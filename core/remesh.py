import torch
import torch.nn.functional as tfunc
import torch_scatter
from typing import Tuple

def prepend_dummies(
        vertices:torch.Tensor, #V,D
        faces:torch.Tensor, #F,3 long
    )->Tuple[torch.Tensor,torch.Tensor]:
    """prepend dummy elements to vertices and faces to enable "masked" scatter operations"""
    # 在顶点和面数据的前面添加虚拟元素，以便支持“掩码”式的 scatter 操作
    
    V,D = vertices.shape
    vertices = torch.concat((torch.full((1,D),fill_value=torch.nan,device=vertices.device),vertices),dim=0)
    # 在顶点张量的前面添加一行全为 NaN 的虚拟顶点
    faces = torch.concat((torch.zeros((1,3),dtype=torch.long,device=faces.device),faces+1),dim=0)
    # 在面张量的前面添加一行全为 0 的虚拟面，并将原有的面索引全部加 1，以适应插入的虚拟顶点
    return vertices,faces

def remove_dummies(
        vertices: torch.Tensor,  # V,D - first vertex all nan and unreferenced
        faces: torch.Tensor,     # F,3 long - first face all zeros
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """remove dummy elements added with prepend_dummies()"""
    # 移除通过 prepend_dummies() 添加的虚拟元素

    return vertices[1:], faces[1:] - 1
    # 删除第一个顶点和第一个面，并将面索引减 1，恢复原始的顶点和面数据


def calc_edges(
        faces: torch.Tensor,  # F,3 long - first face may be dummy with all zeros
        with_edge_to_face: bool = False
    ) -> Tuple[torch.Tensor, ...]:
    """
    returns tuple of 返回一个包含以下内容的元组：
    - edges E,2 long, 0 for unused, lower vertex index first 形状为 (E,2) 的张量，未使用的边用 0 表示，且第一个顶点索引较小
    - face_to_edge F,3 long 形状为 (F,3) 的张量，对应每个面的三条边
    - (optional) edge_to_face shape=E,[left,right],[face,side] （可选）形状为 (E,[left,right],[face,side]) 的张量，表示每条边对应的左右两个面及其在面的哪一侧

    边的表示：
    o-<-----e1     e0,e1...edge, e0<e1 （第一个顶点索引较小）
    |      /A      L,R....left and right face 左侧和右侧的面
    |  L /  |      both triangles ordered counter clockwise 两个三角形按逆时针顺序排列
    |  / R  |      normals pointing out of screen 法线朝向屏幕外
    V/      |      
    e0---->-o     
    """
    
    F = faces.shape[0]
    
    # make full edges, lower vertex index first
    # 创建完整的边，确保每条边的第一个顶点索引较小
    face_edges = torch.stack((faces,faces.roll(-1,1)),dim=-1) #F*3,3,2
    # 生成每个面的三条边，连接顶点索引（c0, c1）、（c1, c2）、（c2, c0）
    full_edges = face_edges.reshape(F*3,2)
    sorted_edges,_ = full_edges.sort(dim=-1) #F*3,2 TODO min/max faster?
    # 对每条边的两个顶点索引进行排序，确保第一个索引较小

    # make unique edges
    # 创建唯一的边集合
    edges,full_to_unique = torch.unique(input=sorted_edges,sorted=True,return_inverse=True,dim=0) #(E,2),(F*3)
    # edges：所有唯一的边；full_to_unique：每个完整边在 edges 中的索引
    E = edges.shape[0]
    face_to_edge = full_to_unique.reshape(F,3) #F,3
    # face_to_edge：每个面对应的三条边在 edges 中的索引

    if not with_edge_to_face:
        return edges, face_to_edge

    # 如果需要 edge_to_face 信息
    is_right = full_edges[:,0]!=sorted_edges[:,0] #F*3
    # 判断每条边在面中的方向，如果排序后顶点不一致，则该边指向右侧
    edge_to_face = torch.zeros((E,2,2),dtype=torch.long,device=faces.device) #E,LR=2,S=2
    # 初始化 edge_to_face 张量，形状为 (E, 2, 2)，分别表示左、右面及其对应的面索引和边索引
    scatter_src = torch.cartesian_prod(torch.arange(0,F,device=faces.device),torch.arange(0,3,device=faces.device)) #F*3,2
    # 生成所有面的边的索引组合
    edge_to_face.reshape(2*E,2).scatter_(dim=0,index=(2*full_to_unique+is_right)[:,None].expand(F*3,2),src=scatter_src) #E,LR=2,S=2
    # 将每条边对应的面和边的信息填充到 edge_to_face 中
    edge_to_face[0] = 0
    return edges, face_to_edge, edge_to_face

def calc_edge_length(
        vertices:torch.Tensor, #V,3 first may be dummy
        edges:torch.Tensor, #E,2 long, lower vertex index first, (0,0) for unused
    )->torch.Tensor: #E
    """Calculate the length of each edge."""
    # 计算每条边的长度

    full_vertices = vertices[edges] #E,2,3
    # 取出每条边的两个顶点坐标
    a,b = full_vertices.unbind(dim=1) #E,3
    # 将顶点坐标解绑定为两个独立的张量
    return torch.norm(a-b,p=2,dim=-1)
    # 计算每条边的欧氏距离，即边的长度

def calc_face_normals(
        vertices:torch.Tensor, #V,3 first vertex may be unreferenced
        faces:torch.Tensor, #F,3 long, first face may be all zero
        normalize:bool=False,
        )->torch.Tensor: #F,3
    """
         n
         |
         c0     corners ordered counterclockwise when
        / \     looking onto surface (in neg normal direction)
      c1---c2
    """
    # 计算每个面的法线向量
    # 顶点顺序按照逆时针排列，当从负法线方向观察表面时
    # 即法线朝向屏幕外，顶点顺序为 c0 -> c1 -> c2
    
    full_vertices = vertices[faces] #F,C=3,3
    # 获取每个面对应的三个顶点坐标
    v0,v1,v2 = full_vertices.unbind(dim=1) #F,3
    # 将顶点坐标解绑定为独立的张量
    face_normals = torch.cross(v1-v0,v2-v0, dim=1) #F,3
    # 计算法线向量，通过两个边向量的叉乘得到
    if normalize:
        face_normals = tfunc.normalize(face_normals, eps=1e-6, dim=1) #TODO inplace?
        # 如果需要规范化，则将法线向量单位化
    return face_normals #F,3
    # 返回面的法线向量

def calc_vertex_normals(
        vertices:torch.Tensor, #V,3 first vertex may be unreferenced
        faces:torch.Tensor, #F,3 long, first face may be all zero
        face_normals:torch.Tensor=None, #F,3, not normalized
    )->torch.Tensor: #F,3
    """Calculate normals for each vertex."""
    # 计算每个顶点的法线向量

    F = faces.shape[0]

    if face_normals is None:
        face_normals = calc_face_normals(vertices,faces)
        # 如果未提供面的法线，则计算面的法线向量
    
    vertex_normals = torch.zeros((vertices.shape[0],3,3),dtype=vertices.dtype,device=vertices.device) #V,C=3,3
    # 初始化顶点法线张量，形状为 (V, 3, 3)，用于累计每个顶点的法线分量
    vertex_normals.scatter_add_(dim=0,index=faces[:,:,None].expand(F,3,3),src=face_normals[:,None,:].expand(F,3,3))
    # 将每个面的法线向量累加到对应的顶点上
    vertex_normals = vertex_normals.sum(dim=1) #V,3
    # 对每个顶点的法线分量求和  
    return tfunc.normalize(vertex_normals, eps=1e-6, dim=1)
    # 返回单位化的顶点法线向量

def calc_face_ref_normals(
        faces:torch.Tensor, #F,3 long, 0 for unused
        vertex_normals:torch.Tensor, #V,3 first unused
        normalize:bool=False,
    )->torch.Tensor: #F,3
    """calculate reference normals for face flip detection"""
    # 计算参考法线，用于检测面的翻转
    
    full_normals = vertex_normals[faces] #F,C=3,3
    # 获取每个面对应的三个顶点的法线向量
    ref_normals = full_normals.sum(dim=1) #F,3
    # 对三个顶点的法线向量求和，作为面的参考法线
    if normalize:
        ref_normals = tfunc.normalize(ref_normals, eps=1e-6, dim=1)
        # 如果需要规范化，则将参考法线单位化
    return ref_normals

def pack(
        vertices:torch.Tensor, #V,3 first unused and nan
        faces:torch.Tensor, #F,3 long, 0 for unused
        )->Tuple[torch.Tensor,torch.Tensor]: #(vertices,faces), keeps first vertex unused
    """removes unused elements in vertices and faces"""
    # 移除顶点和面中未使用的元素
    
    V = vertices.shape[0]
    
    # remove unused faces
    # 移除未使用的面
    used_faces = faces[:,0]!=0
    used_faces[0] = True
    faces = faces[used_faces] #sync
    # 保留第一个面，以及所有第一个顶点索引不为 0 的面

    # remove unused vertices
    # 移除未使用的顶点
    used_vertices = torch.zeros(V,3,dtype=torch.bool,device=vertices.device)
    used_vertices.scatter_(dim=0,index=faces,value=True,reduce='add') #TODO int faster?
    # 标记在 faces 中出现过的顶点为使用状态
    used_vertices = used_vertices.any(dim=1)
    used_vertices[0] = True
    vertices = vertices[used_vertices] #sync
    # 保留第一个顶点，以及所有被使用过的顶点

    # update used faces
    # 更新面的顶点索引
    ind = torch.zeros(V,dtype=torch.long,device=vertices.device)
    V1 = used_vertices.sum()
    ind[used_vertices] =  torch.arange(0,V1,device=vertices.device) #sync
    faces = ind[faces]
    # 将 faces 中的顶点索引更新为新的索引

    return vertices,faces

def split_edges(
        vertices:torch.Tensor, #V,3 first unused
        faces:torch.Tensor, #F,3 long, 0 for unused
        edges:torch.Tensor, #E,2 long 0 for unused, lower vertex index first
        face_to_edge:torch.Tensor, #F,3 long 0 for unused
        splits, #E bool
        pack_faces:bool=True,
        )->Tuple[torch.Tensor,torch.Tensor]: #(vertices,faces)

    # 边的细分：
    #   c2                    c2               c...corners = faces 顶点 = faces
    #    . .                   . .             s...side_vert, 0 means no split 侧边顶点，0 表示不细分
    #    .   .                 .N2 .           S...shrunk_face 缩小的面
    #    .     .               .     .         Ni...new_faces 新面
    #   s2      s1           s2|c2...s1|c1
    #    .        .            .     .  .
    #    .          .          . S .      .
    #    .            .        . .     N1    .
    #   c0...(s0=0)....c1    s0|c0...........c1
    #
    # pseudo-code 伪代码:
    #   S = [s0|c0,s1|c1,s2|c2] example:[c0,s1,s2]
    #   split = side_vert!=0 example:[False,True,True]
    #   N0 = split[0]*[c0,s0,s2|c2] example:[0,0,0]
    #   N1 = split[1]*[c1,s1,s0|c0] example:[c1,s1,c0]
    #   N2 = split[2]*[c2,s2,s1|c1] example:[c2,s2,s1]

    V = vertices.shape[0]
    F = faces.shape[0]
    S = splits.sum().item() #sync
    # 统计需要细分的边的数量

    if S==0:
        return vertices,faces
        # 如果没有需要细分的边，则直接返回
    
    edge_vert = torch.zeros_like(splits, dtype=torch.long) #E
    # 为每条需要细分的边分配新的顶点索引
    edge_vert[splits] = torch.arange(V,V+S,dtype=torch.long,device=vertices.device) #E 0 for no split, sync
    side_vert = edge_vert[face_to_edge] #F,3 long, 0 for no split
    # 获取每个面对应的三条边的新顶点索引，未细分的边为 0
    split_edges = edges[splits] #S sync
    # 获取需要细分的边

    #vertices
    # 更新顶点
    split_vertices = vertices[split_edges].mean(dim=1) #S,3
    # 计算细分边中点的位置，作为新的顶点
    vertices = torch.concat((vertices,split_vertices),dim=0)
    # 将新的顶点添加到顶点列表中

    #faces
    # 更新面
    side_split = side_vert!=0 #F,3
    # 标记哪些边进行了细分
    shrunk_faces = torch.where(side_split,side_vert,faces) #F,3 long, 0 for no split
    # 使用新的顶点替换细分边对应的顶点
    new_faces = side_split[:,:,None] * torch.stack((faces,side_vert,shrunk_faces.roll(1,dims=-1)),dim=-1) #F,N=3,C=3
    # 构建新的面，包含新的三角形
    faces = torch.concat((shrunk_faces,new_faces.reshape(F*3,3))) #4F,3
    # 将缩小的旧面和新面合并
    if pack_faces:
        mask = faces[:,0]!=0
        mask[0] = True
        faces = faces[mask] #F',3 sync
        # 移除无效的面

    return vertices,faces

def collapse_edges(
        vertices:torch.Tensor, #V,3 first unused
        faces:torch.Tensor, #F,3 long 0 for unused
        edges:torch.Tensor, #E,2 long 0 for unused, lower vertex index first
        priorities:torch.Tensor, #E float
        lock_index:int=-1, # Points with index < lock_index will not be deleted
        stable:bool=False, #only for unit testing
        )->Tuple[torch.Tensor,torch.Tensor]: #(vertices,faces)
    """Collapse edges based on priorities."""
    # 根据优先级对边进行折叠

    V = vertices.shape[0]
    
    # check spacing
    # 计算边的排序
    _,order = priorities.sort(stable=stable) #E
    rank = torch.zeros_like(order)
    rank[order] = torch.arange(0,len(rank),device=rank.device)
    # 为每条边分配排名
    vert_rank = torch.zeros(V,dtype=torch.long,device=vertices.device) #V
    edge_rank = rank #E
    
    # 迭代更新顶点的排名
    for i in range(3):
        torch_scatter.scatter_max(src=edge_rank[:,None].expand(-1,2).reshape(-1),index=edges.reshape(-1),dim=0,out=vert_rank)
        edge_rank,_ = vert_rank[edges].max(dim=-1) #E
    candidates = edges[(edge_rank==rank).logical_and_(priorities>0)] #E',2
    # 选择符合条件的边进行折叠

    # check connectivity
    # 检查连接性
    vert_connections = torch.zeros(V,dtype=torch.long,device=vertices.device) #V
    vert_connections[candidates[:,0]] = 1 #start 标记起始顶点
    edge_connections = vert_connections[edges].sum(dim=-1) #E, edge connected to start
    vert_connections.scatter_add_(dim=0,index=edges.reshape(-1),src=edge_connections[:,None].expand(-1,2).reshape(-1))# one edge from start
    vert_connections[candidates] = 0 #clear start and end
    edge_connections = vert_connections[edges].sum(dim=-1) #E, one or two edges from start
    vert_connections.scatter_add_(dim=0,index=edges.reshape(-1),src=edge_connections[:,None].expand(-1,2).reshape(-1)) #one or two edges from start
    collapses = candidates[vert_connections[candidates[:,1]] <= 2] # E" not more than two connections between start and end
    # 选择不会导致非流形的边进行折叠
    
    # 筛选可以折叠的边
    if lock_index >= 0:
        valid_collapses = (collapses[:, 0] >= lock_index) & (collapses[:, 1] >= lock_index)
        collapses = collapses[valid_collapses]
    
    if len(collapses) == 0:
        return vertices, faces

    # 处理一个点是锁定点，一个不是的情况
    if lock_index >= 0:
        locked_mask = collapses < lock_index
        any_locked = locked_mask.any(dim=1)
        
        # 对于每个边，如果有一个点被锁定，我们保留锁定的点
        collapses[any_locked, 0] = torch.where(locked_mask[any_locked, 0], 
                                               collapses[any_locked, 0], 
                                               collapses[any_locked, 1])
        collapses[any_locked, 1] = torch.where(locked_mask[any_locked, 0], 
                                               collapses[any_locked, 1], 
                                               collapses[any_locked, 0])
    
    # mean vertices
    # 更新顶点位置
    vertices[collapses[:,0]] = vertices[collapses].mean(dim=1) #TODO dim?
    # 将折叠边的两个顶点位置取平均，作为新的顶点位置
    
    # update faces
    # 更新面信息
    dest = torch.arange(0, V, dtype=torch.long, device=vertices.device) #V
    dest[collapses[:,1]] = dest[collapses[:,0]]
    faces = dest[faces] #F,3 TODO optimize?
    c0,c1,c2 = faces.unbind(dim=-1)
    collapsed = (c0==c1).logical_or_(c1==c2).logical_or_(c0==c2)
    faces[collapsed] = 0
    # 删除退化的面

    return vertices,faces

def calc_face_collapses(
        vertices:torch.Tensor, #V,3 first unused
        faces:torch.Tensor, #F,3 long, 0 for unused
        edges:torch.Tensor, #E,2 long 0 for unused, lower vertex index first
        face_to_edge:torch.Tensor, #F,3 long 0 for unused
        edge_length:torch.Tensor, #E
        face_normals:torch.Tensor, #F,3
        vertex_normals:torch.Tensor, #V,3 first unused
        min_edge_length:torch.Tensor=None, #V
        area_ratio = 0.5, #collapse if area < min_edge_length**2 * area_ratio
        shortest_probability = 0.8
    )->torch.Tensor: #E edges to collapse
    """Determine which faces to collapse based on various criteria."""
    # 根据各种标准确定需要折叠的面
    
    E = edges.shape[0]
    F = faces.shape[0]

    # face flips
    # 检测面法线翻转
    ref_normals = calc_face_ref_normals(faces,vertex_normals,normalize=False) #F,3
    face_collapses = (face_normals*ref_normals).sum(dim=-1)<0 #F
    # 如果面的法线与参考法线方向相反，则标记为需要折叠
    
    # small faces
    # 处理小面
    if min_edge_length is not None:
        min_face_length = min_edge_length[faces].mean(dim=-1) #F
        min_area = min_face_length**2 * area_ratio #F
        face_collapses.logical_or_(face_normals.norm(dim=-1) < min_area*2) #F
        face_collapses[0] = False
        # 标记面积过小的面进行折叠

    # faces to edges
    # 将面的折叠情况映射到边
    face_length = edge_length[face_to_edge] #F,3

    if shortest_probability<1:
        #select shortest edge with shortest_probability chance
        # 根据概率选择要折叠的边
        randlim = round(2/(1-shortest_probability))
        rand_ind = torch.randint(0,randlim,size=(F,),device=faces.device).clamp_max_(2) #selected edge local index in face
        sort_ind = torch.argsort(face_length,dim=-1,descending=True) #F,3
        local_ind = sort_ind.gather(dim=-1,index=rand_ind[:,None])
    else:
        local_ind = torch.argmin(face_length,dim=-1)[:,None] #F,1 0...2 shortest edge local index in face
    
    edge_ind = face_to_edge.gather(dim=1,index=local_ind)[:,0] #F 0...E selected edge global index
    edge_collapses = torch.zeros(E,dtype=torch.long,device=vertices.device)
    edge_collapses.scatter_add_(dim=0,index=edge_ind,src=face_collapses.long()) #TODO legal for bool?
    # 统计需要折叠的边

    return edge_collapses.bool()

def flip_edges(
        vertices:torch.Tensor, #V,3 first unused
        faces:torch.Tensor, #F,3 long, first must be 0, 0 for unused
        edges:torch.Tensor, #E,2 long, first must be 0, 0 for unused, lower vertex index first
        edge_to_face:torch.Tensor, #E,[left,right],[face,side]
        with_border:bool=True, #handle border edges (D=4 instead of D=6)
        with_normal_check:bool=True, #check face normal flips
        stable:bool=False, #only for unit testing
    ):
    """Flip edges to improve mesh quality."""
    # 翻转边以改进网格质量
    
    V = vertices.shape[0]
    E = edges.shape[0]
    device=vertices.device
    vertex_degree = torch.zeros(V,dtype=torch.long,device=device) #V long
    vertex_degree.scatter_(dim=0,index=edges.reshape(E*2),value=1,reduce='add')
    # 计算每个顶点的度（连接的边数）
    neighbor_corner = (edge_to_face[:,:,1] + 2) % 3 #go from side to corner
    # 计算相邻顶点的索引
    neighbors = faces[edge_to_face[:,:,0],neighbor_corner] #E,LR=2
    edge_is_inside = neighbors.all(dim=-1) #E
    # 判断边是否是内部边（两侧都有面）

    if with_border:
        # 处理边界情况
        # inside vertices should have D=6, border edges D=4, so we subtract 2 for all inside vertices
        # need to use float for masks in order to use scatter(reduce='multiply')
        vertex_is_inside = torch.ones(V,2,dtype=torch.float32,device=vertices.device) #V,2 float
        src = edge_is_inside.type(torch.float32)[:,None].expand(E,2) #E,2 float
        vertex_is_inside.scatter_(dim=0,index=edges,src=src,reduce='multiply')
        vertex_is_inside = vertex_is_inside.prod(dim=-1,dtype=torch.long) #V long
        vertex_degree -= 2 * vertex_is_inside #V long
        # 内部顶点的度减 2

    neighbor_degrees = vertex_degree[neighbors] #E,LR=2
    edge_degrees = vertex_degree[edges] #E,2
    # 计算邻居和边顶点的度
    #
    # loss = Sum_over_affected_vertices((new_degree-6)**2)
    # loss_change = Sum_over_neighbor_vertices((degree+1-6)**2-(degree-6)**2)
    #                   + Sum_over_edge_vertices((degree-1-6)**2-(degree-6)**2)
    #             = 2 * (2 + Sum_over_neighbor_vertices(degree) - Sum_over_edge_vertices(degree))
    #
    # 计算翻转后度的变化
    loss_change = 2 + neighbor_degrees.sum(dim=-1) - edge_degrees.sum(dim=-1) #E
    candidates = torch.logical_and(loss_change<0, edge_is_inside) #E
    loss_change = loss_change[candidates] #E'
    if loss_change.shape[0]==0:
        return

    edges_neighbors = torch.concat((edges[candidates],neighbors[candidates]),dim=-1) #E',4
    _,order = loss_change.sort(descending=True, stable=stable) #E'
    rank = torch.zeros_like(order)
    rank[order] = torch.arange(0,len(rank),device=rank.device)
    vertex_rank = torch.zeros((V,4),dtype=torch.long,device=device) #V,4
    torch_scatter.scatter_max(src=rank[:,None].expand(-1,4),index=edges_neighbors,dim=0,out=vertex_rank)
    vertex_rank,_ = vertex_rank.max(dim=-1) #V
    neighborhood_rank,_ = vertex_rank[edges_neighbors].max(dim=-1) #E'
    flip = rank==neighborhood_rank #E'

    if with_normal_check:
        #  cl-<-----e1     e0,e1...edge, e0<e1
        #   |      /A      L,R....left and right face
        #   |  L /  |      both triangles ordered counter clockwise
        #   |  / R  |      normals pointing out of screen
        #   V/      |      
        #   e0---->-cr    
        # 检查翻转后是否会导致法线翻转
        v = vertices[edges_neighbors] #E",4,3
        v = v - v[:,0:1] # make relative to e0 以 e0 为原点
        e1 = v[:,1]
        cl = v[:,2]
        cr = v[:,3]
        n = torch.cross(e1,cl) + torch.cross(cr,e1) #sum of old normal vectors 
        flip.logical_and_(torch.sum(n*torch.cross(cr,cl),dim=-1)>0) #first new face
        flip.logical_and_(torch.sum(n*torch.cross(cl-e1,cr-e1),dim=-1)>0) #second new face

    flip_edges_neighbors = edges_neighbors[flip] #E",4
    flip_edge_to_face = edge_to_face[candidates,:,0][flip] #E",2
    flip_faces = flip_edges_neighbors[:,[[0,3,2],[1,2,3]]] #E",2,3
    faces.scatter_(dim=0,index=flip_edge_to_face.reshape(-1,1).expand(-1,3),src=flip_faces.reshape(-1,3))
    # 更新 faces，完成边的翻转
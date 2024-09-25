from copy import deepcopy
import time
import torch
import torch_scatter
from core.remesh import (
    calc_edges,
    remesh,
    split_edges,
    pack
)
from typing import Tuple

def lerp_unbiased(a: torch.Tensor, b: torch.Tensor, weight: float, step: int):
    """lerp with adam's bias correction"""
    c_prev = 1 - weight**(step - 1)
    c = 1 - weight**step
    a_weight = weight * c_prev / c
    b_weight = (1 - weight) / c
    a.mul_(a_weight).add_(b, alpha=b_weight)  # 进行线性插值，并进行偏差校正

class MeshOptimizer:
    """Use this like a pytorch Optimizer, but after calling opt.step(), do vertices,faces = opt.remesh()."""

    def __init__(
        self,
        vertices: torch.Tensor,  # V,3
        faces: torch.Tensor,     # F,3
        lock_faces_index: torch.Tensor = None,  # 新增参数，用于指定锁定的面索引
        lr=0.3,                  # learning rate
        betas=(0.8, 0.8, 0),     # betas[0:2] are the same as in Adam, betas[2] may be used to time-smooth the relative velocity nu
        gammas=(0, 0, 0),        # optional spatial smoothing for m1,m2,nu, values between 0 (no smoothing) and 1 (max. smoothing)
        nu_ref=0.3,              # reference velocity for edge length controller
        edge_len_lims=(.01, .15),# smallest and largest allowed reference edge length
        edge_len_tol=.5,         # edge length tolerance for split and collapse
        gain=.2,                 # gain value for edge length controller
        laplacian_weight=.02,    # for laplacian smoothing/regularization
        ramp=1,                  # learning rate ramp, actual ramp width is ramp/(1-betas[0])
        grad_lim=10.,            # gradients are clipped to m1.abs()*grad_lim
        remesh_interval=1,       # larger intervals are faster but with worse mesh quality
        local_edgelen=True,      # set to False to use a global scalar reference edge length instead
        warmup_steps=10,          # 新增：预热步数
        warmup_factor=0.1,       # 新增：预热因子，表示初始变化速率相对于正常速率的比例
    ):
        self._vertices = vertices  # 存储当前顶点
        self._faces = faces        # 存储当前面
        self._lock_faces_index = lock_faces_index  # 存储锁定的面索引
        self._lr = lr              # 学习率
        self._betas = betas        # 动量参数
        self._gammas = gammas      # 平滑参数warmup_steps
        self._nu_ref = nu_ref      # 参考速度
        self._edge_len_lims = edge_len_lims  # 边长限制
        self._edge_len_tol = edge_len_tol      # 边长容差
        self._gain = gain                    # 增益值
        self._laplacian_weight = laplacian_weight  # 拉普拉斯平滑权重
        self._ramp = ramp                    # 学习率调整参数
        self._grad_lim = grad_lim            # 梯度裁剪限幅
        self._remesh_interval = remesh_interval  # 网格重划分间隔
        self._local_edgelen = local_edgelen      # 是否使用局部边长
        self._step = 0                        # 当前优化步数
        self._start = time.time()             # 记录优化开始时间
        self._warmup_steps = warmup_steps     # 预热步数
        self._warmup_factor = warmup_factor   # 预热因子
        self._current_warmup_factor = warmup_factor  # 当前预热因子

        V = self._vertices.shape[0]  # 顶点数量
        self._initial_vertex_count = vertices.shape[0]  # 存储初始顶点数量
        # 为所有基于顶点的数据准备连续的张量
        self._vertices_etc = torch.zeros([V, 9], device=vertices.device)  # 初始化附加顶点信息张量
        self._split_vertices_etc()  # 分割附加顶点信息
        self.vertices.copy_(vertices)  # 初始化顶点
        self._vertices.requires_grad_()  # 设置顶点需要梯度
        self._ref_len.fill_(edge_len_lims[1])  # 初始化参考边长为最大边长限制
    
    @torch.no_grad()
    def initial_subdivision(self):
        # 计算当前的边和面到边的映射
        edges, face_to_edge = calc_edges(self._faces)
        
        # 创建一个布尔张量，表示对所有的边进行细分
        splits = torch.ones(edges.shape[0], dtype=torch.bool, device=edges.device)
        
        # 进行边的细分操作
        self._vertices_etc, self._faces = split_edges(
            self._vertices_etc,
            self._faces,
            edges,
            face_to_edge,
            splits,
            lock_faces_index=None  # 这里不需要锁定面，因为我们要细分所有边
        )
        
        # 移除未使用的顶点和面，整理数据，保留初始顶点的索引
        self._vertices_etc, self._faces = pack(
            self._vertices_etc,
            self._faces,
            initial_vertex_count=self._initial_vertex_count+1  # 保留初始顶点索引
        )
        
        # 更新内部变量
        self._split_vertices_etc()
        self._vertices.requires_grad_()

    @property
    def vertices(self):
        return self._vertices  # 返回当前顶点

    @property
    def faces(self):
        return self._faces  # 返回当前面

    def _split_vertices_etc(self):
        self._vertices = self._vertices_etc[:, :3]       # 提取顶点坐标
        self._m2 = self._vertices_etc[:, 3]             # 提取动量 m2
        self._nu = self._vertices_etc[:, 4]             # 提取速度 nu
        self._m1 = self._vertices_etc[:, 5:8]           # 提取动量 m1
        self._ref_len = self._vertices_etc[:, 8]        # 提取参考边长

        with_gammas = any(g != 0 for g in self._gammas)  # 检查是否需要空间平滑
        self._smooth = self._vertices_etc[:, :8] if with_gammas else self._vertices_etc[:, :3]  # 设置平滑数据

    def zero_grad(self):
        self._vertices.grad = None  # 清除顶点梯度

    @torch.no_grad()
    def step(self):
        eps = 1e-8  # 防止除零错误

        self._step += 1  # 增加优化步数

        # 空间平滑
        edges, _ = calc_edges(self._faces)  # 计算当前面的所有边
        E = edges.shape[0]  # 边的数量
        edge_smooth = self._smooth[edges]  # 获取边对应的平滑数据
        neighbor_smooth = torch.zeros_like(self._smooth)  # 初始化邻居平滑数据张量
        torch_scatter.scatter_mean(
            src=edge_smooth.flip(dims=[1]).reshape(E * 2, -1),
            index=edges.reshape(E * 2, 1),
            dim=0,
            out=neighbor_smooth,
        )  # 计算每个顶点的邻居平滑平均值

        # 应用可选的 m1, m2, nu 的空间平滑
        if self._gammas[0]:
            self._m1.lerp_(neighbor_smooth[:, 5:8], self._gammas[0])  # 平滑动量 m1
        if self._gammas[1]:
            self._m2.lerp_(neighbor_smooth[:, 3], self._gammas[1])    # 平滑动量 m2
        if self._gammas[2]:
            self._nu.lerp_(neighbor_smooth[:, 4], self._gammas[2])    # 平滑速度 nu

        # 添加拉普拉斯平滑到梯度
        laplace = self._vertices - neighbor_smooth[:, :3]  # 计算拉普拉斯平滑项
        grad = torch.addcmul(
            self._vertices.grad, laplace, self._nu[:, None], value=self._laplacian_weight
        )  # 将拉普拉斯平滑项添加到梯度中

        # 梯度裁剪
        if self._step > 1:
            grad_lim = self._m1.abs().mul_(self._grad_lim)  # 计算梯度裁剪的限制
            grad.clamp_(min=-grad_lim, max=grad_lim)      # 对梯度进行裁剪

        # 动量更新
        lerp_unbiased(self._m1, grad, self._betas[0], self._step)  # 更新动量 m1
        lerp_unbiased(self._m2, (grad**2).sum(dim=-1), self._betas[1], self._step)  # 更新动量 m2

        velocity = self._m1 / self._m2[:, None].sqrt().add_(eps)  # 计算速度 velocity
        speed = velocity.norm(dim=-1)  # 计算速度的范数 speed

        if self._betas[2]:
            lerp_unbiased(self._nu, speed, self._betas[2], self._step)  # 平滑速度 nu
        else:
            self._nu.copy_(speed)  # 直接复制速度 nu

        # 更新顶点位置
        ramped_lr = self._lr * min(1, self._step * (1 - self._betas[0]) / self._ramp)  # 计算调整后的学习率

        # 在更新顶点位置之前应用预热因子
        if self._step <= self._warmup_steps:
            warmup_lr = ramped_lr * self._current_warmup_factor
            self._vertices.add_(velocity * self._ref_len[:, None], alpha=-warmup_lr)
            
            # 更新预热因子
            progress = self._step / self._warmup_steps
            self._current_warmup_factor = self._warmup_factor + (1 - self._warmup_factor) * progress
        else:
            self._vertices.add_(velocity * self._ref_len[:, None], alpha=-ramped_lr)

        # 更新目标边长
        if self._step % self._remesh_interval == 0:
            if self._local_edgelen:
                len_change = 1 + (self._nu - self._nu_ref) * self._gain  # 局部边长变化
            else:
                len_change = 1 + (self._nu.mean() - self._nu_ref) * self._gain  # 全局边长变化
            self._ref_len *= len_change  # 更新参考边长
            self._ref_len.clamp_(*self._edge_len_lims)  # 限制参考边长在指定范围内

    def remesh(self, flip: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        min_edge_len = self._ref_len * (1 - self._edge_len_tol)  # 计算最小边长限制
        max_edge_len = self._ref_len * (1 + self._edge_len_tol)  # 计算最大边长限制

        self._vertices_etc, self._faces, self._faces_mapping = remesh(
            self._vertices_etc, 
            self._faces, 
            min_edge_len, 
            max_edge_len, 
            flip,
            lock_index=self._initial_vertex_count,  # 使用初始顶点数量作为 lock_index
            lock_faces_index=self._lock_faces_index,  # 传递锁定的面索引
        )  # 进行网格重划分

        # 更新锁定的面索引
        if self._lock_faces_index is not None:
            self._lock_faces_index = self._update_lock_faces_index()

        self._split_vertices_etc()  # 分割更新后的顶点信息
        self._vertices.requires_grad_()  # 设置顶点需要梯度

        return self._vertices, self._faces  # 返回更新后的顶点和面

    # 新增函数，用于更新锁定的面索引
    def _update_lock_faces_index(self):
        return self._faces_mapping[self._lock_faces_index]
from core.remesh import calc_vertex_normals  # 从 core.remesh 模块中导入计算顶点法线的函数
from core.opt import MeshOptimizer           # 从 core.opt 模块中导入网格优化器类
from util.func import load_obj, make_sphere, make_circular_cameras, normalize_vertices, save_obj, save_images  # 导入实用函数
from util.render import NormalsRenderer      # 导入法线渲染器类
from tqdm import tqdm                        # 导入 tqdm 库用于显示进度条
from util.snapshot import snapshot           # 导入快照函数
from copy import deepcopy

import os
import torch

# 获取当前 GPU 的计算能力
capability = torch.cuda.get_device_capability()
arch = f'{capability[0]}.{capability[1]}'
print(f"CUDA 计算能力：{arch}")

# 设置环境变量
os.environ['TORCH_CUDA_ARCH_LIST'] = arch

# 尝试导入显示函数，如果失败则设置为 None
try:
    from util.view import show               # 从 util.view 模块中导入显示函数
    print("成功导入显示函数 'show'")
except ImportError:
    show = None                              # 如果导入失败，将 show 设置为 None
    print("无法导入显示函数 'show'，设为 None")

# 设置文件名和参数
fname = 'data/makehuman_tgt_f.obj'                      # 目标模型的文件路径
steps = 150                                # 优化的总步数
snapshot_step = 1                            # 每隔多少步保存一次快照
print(f"目标文件：{fname}")
print(f"优化步骤数：{steps}")
print(f"快照间隔：{snapshot_step}")

# 创建摄像机视角
# mv, proj = make_star_cameras(4, 4)           # 创建一个 4x4 的星型摄像机矩阵
mv, proj = make_circular_cameras(6)           # 创建一个 6 个摄像机的圆形摄像机矩阵
print("创建摄像机视角完成")

renderer = NormalsRenderer(mv, proj, [512, 512])  # 初始化法线渲染器，设置视图矩阵、投影矩阵和图像尺寸
print("法线渲染器初始化完成")

# 加载并处理目标模型
print("开始加载目标模型")
target_vertices, target_faces = load_obj(fname)     # 加载目标模型的顶点和面数据
print(f"目标模型顶点数量：{target_vertices.shape[0]}")
print(f"目标模型面数量：{target_faces.shape[0]}")

target_vertices = normalize_vertices(target_vertices)  # 对顶点进行归一化处理
print("目标模型顶点归一化完成")

target_normals = calc_vertex_normals(target_vertices, target_faces)  # 计算目标模型的顶点法线
print("计算目标模型顶点法线完成")

target_images = renderer.render(target_vertices, target_normals, target_faces)  # 渲染目标模型，得到目标图像
print("渲染目标模型完成")

save_images(target_images[..., :3], './out/target_images/')        # 保存目标图像的 RGB 通道
save_images(target_images[..., 3:], './out/target_alpha/')         # 保存目标图像的 Alpha 通道
print("目标图像保存完成")

# 初始化待优化的模型
print("初始化待优化的模型")
# vertices, faces = make_sphere(level=2, radius=0.5)  # 创建一个初始的球形网格模型
vertices, faces = load_obj('data/makehuman_base.obj')  # 加载基础人体模型
vertices = normalize_vertices(vertices)  # 对顶点进行归一化处理
init_vert_num = vertices.shape[0]
init_face = deepcopy(faces)
print(f"初始模型顶点数量：{vertices.shape[0]}")
print(f"初始模型面数量：{faces.shape[0]}")

opt = MeshOptimizer(vertices, faces)    # 初始化网格优化器对象
print("网格优化器初始化完成")

# 执行初始细分
# opt.initial_subdivision()

vertices = opt.vertices  # 获取更新后的顶点数据
faces = opt.faces        # 获取更新后的面数据
# print(f"初始细分后的顶点数量：{vertices.shape[0]}")
# print(f"初始细分后的面数量：{faces.shape[0]}")
# save_obj(vertices[:init_vert_num], init_face, './out/result_sub_init.obj')
# save_obj(vertices, faces, './out/result_sub.obj')

snapshots = []                          # 初始化快照列表，用于保存中间结果

# 开始优化过程
print("开始优化过程")
for i in tqdm(range(steps), desc="优化进度", unit="步"):
    opt.zero_grad()                     # 清除前一次迭代的梯度
    normals = calc_vertex_normals(vertices, faces)    # 计算当前模型的顶点法线
    images = renderer.render(vertices, normals, faces)  # 渲染当前模型，得到图像
    
    # # 保存中间图像
    # save_images(images[..., :3], f'./out/intermediate_images/step_{i:04d}/')
    # save_images(images[..., 3:], f'./out/intermediate_alpha/step_{i:04d}/')
    
    loss = (images - target_images).abs().mean()     # 计算当前模型与目标模型的差异，得到损失值
    loss_value = loss.item()
    tqdm.write(f"步骤 {i+1}/{steps}，损失值：{loss_value:.6f}")

    loss.backward()                   # 反向传播，计算梯度
    opt.step()                        # 更新模型参数

    # 每隔指定步数保存一次快照
    if show and i % snapshot_step == 0:
        snapshots.append(snapshot(opt))  # 保存当前模型的快照
    vertices, faces = opt.remesh()    # 进行重新网格化，更新顶点和面数据
    tqdm.write(f"重新网格化：顶点数量 {vertices.shape[0]}，面数量 {faces.shape[0]}")

print("优化过程完成")

# 保存优化后的模型和结果图像
save_obj(vertices, faces, './out/result.obj')             # 保存最终的模型为 OBJ 文件
print("保存优化后的模型为 'result.obj'")

save_obj(vertices[:init_vert_num], init_face, './out/result_init.obj')             # 保存最终的初始点模型为 OBJ 文件
print("保存优化后的初始点模型为 'result_init.obj'")

save_images(images[..., :3], './out/images/')             # 保存最终的 RGB 渲染图像
save_images(images[..., 3:], './out/alpha/')              # 保存最终的 Alpha 通道图像
print("保存优化后的图像")

# 如果支持显示功能，展示目标模型和优化过程中的模型
if show:
    print("显示优化结果")
    show(target_vertices, target_faces, snapshots)        # 显示目标模型和优化过程
else:
    print("显示功能不可用")
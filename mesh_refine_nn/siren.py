import torch
import torch.nn as nn
import torch.optim as optim
import math
from Lossfunction import compute_face_normals,angle_constraint_loss,total_variation_loss
from ValleyEdgeDetector import EdgeFacerIntersectionDetector



class Sine(nn.Module):
    """SIREN 的激活：y = sin(omega_0 * x)"""
    def __init__(self, omega_0=1.0):
        super().__init__()
        self.omega_0 = omega_0

    def forward(self, x):
        return torch.sin(self.omega_0 * x)


def siren_init_(linear: nn.Linear, omega_0: float, is_first: bool):
    """
    按 SIREN 论文的建议做初始化：
      - 第一层: U(-1/in, 1/in)
      - 之后层: U(-sqrt(6/in)/omega_0, sqrt(6/in)/omega_0)
    """
    in_features = linear.in_features
    with torch.no_grad():
        if is_first:
            bound = 1.0 / in_features
        else:
            bound = math.sqrt(6.0 / in_features) / omega_0
        linear.weight.uniform_(-bound, bound)
        if linear.bias is not None:
            linear.bias.uniform_(-bound, bound)


class SIRENLayer(nn.Module):
    def __init__(self, in_features, out_features, omega_0=1.0, is_first=False, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.activation = Sine(omega_0)
        siren_init_(self.linear, omega_0=omega_0, is_first=is_first)

    def forward(self, x):
        return self.activation(self.linear(x))


class MeshOptimizerSIREN(nn.Module):
    """
    用 SIREN 的三维网格优化网络
    输入: 顶点坐标 [N, 3]
    输出: 优化后的顶点坐标 [N, 3] = x + delta
    """
    def __init__(self, input_size=3, hidden_size=256, hidden_layers=2, w0=30.0, w0_hidden=1.0):
        """
        参数:
          input_size: 输入维度 (顶点坐标=3)
          hidden_size: 每层通道数
          hidden_layers: 隐藏层层数（不含首层）
          w0: 第一层的 omega_0（推荐 30）
          w0_hidden: 隐藏层的 omega_0（推荐 1）
        """
        super().__init__()
        layers = []
        # 第一层：较大的 w0 有助于捕获高频
        layers.append(SIRENLayer(input_size, hidden_size, omega_0=w0, is_first=True))
        # 隐藏层
        for _ in range(hidden_layers):
            layers.append(SIRENLayer(hidden_size, hidden_size, omega_0=w0_hidden, is_first=False))
        self.net = nn.Sequential(*layers)

        # 最后一层线性映射到坐标增量 (不加激活，论文里常见做法)
        self.final = nn.Linear(hidden_size, 3)
        # 输出层也做一个较小范围的初始化，避免初始形变过大
        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden_size) / w0_hidden
            self.final.weight.uniform_(-bound, bound)
            if self.final.bias is not None:
                self.final.bias.zero_()

    def forward(self, x):
        h = self.net(x)
        delta = self.final(h)
        return x + delta
    
def train_mesh_optimization(original_vertices, faces, min_angle_deg=30, num_epochs=1000):
    """
    训练网格优化模型
    
    参数:
    original_vertices: [N, 3] 原始顶点坐标
    faces: [M, 3] 面索引
    min_angle_deg: 最小允许角度(度)
    num_epochs: 训练轮数
    
    返回:
    optimized_vertices: 优化后的顶点坐标
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print('device:',device)
    # 转换为PyTorch张量
    #print(original_vertices)
    original_vertices = torch.tensor(original_vertices, dtype=torch.float32).to(device)
    faces = torch.tensor(faces, dtype=torch.long).to(device)
    #print(original_vertices)
 
    # 初始化模型
    # 初始化模型（SIREN 建议把学习率稍微调低一点）
    model = MeshOptimizerSIREN(
        input_size=3,
        hidden_size=256,
        hidden_layers=2,   # 你可以增减这个深度；2~4 都常见
        w0=30.0,
        w0_hidden=1.0
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  # 从 1e-3 调到 1e-4 更稳
    
    # 损失权重
    w_angle = 7.0  # 角度约束权重
    w_intersect = 10  # 自交约束权重
    w_valley = 5
    w_smooth = 0.01  # 平滑约束权重
    w_deform = 10.0  # 形变最小化权重
    
    detector = EdgeFacerIntersectionDetector(faces,original_vertices)
    # 训练循环
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # 前向传播 - 获取优化后的顶点
        optimized_vertices = model(original_vertices)
        
        # 计算各项损失
        detector.compute_triangle_normals(optimized_vertices)

        # 1. 法向量角度约束
        normals = compute_face_normals(optimized_vertices, faces)
        angle_loss = angle_constraint_loss(normals, min_angle_deg)
        
        # 2. 自交约束 (简化版)
        intersect_loss = detector.compute_intersection_loss()
        #intersect_loss = self_intersection_penalty(optimized_vertices, faces)
        
        # 3. 平滑约束
        smooth_loss = total_variation_loss(optimized_vertices, faces)
        
        # 4. 形变最小化 (原始坐标与优化坐标的距离)
        deform_loss = torch.mean(torch.norm(optimized_vertices - original_vertices, dim=1))

        # 5. 山谷角度损失
            # 检测山谷边
        #valley_flag, _, _ = detector.compute_valley_edges(optimized_vertices)
        
        # 计算坡度角度损失
        slope_loss, angles = detector.compute_slope_angle_loss_all_edges(
           
            
            A_deg=min_angle_deg
        )
        
        # 总损失
        total_loss = (w_angle * angle_loss + 
                     w_intersect * intersect_loss + 
                     w_smooth * smooth_loss + 
                     w_deform * deform_loss +
                     w_valley * slope_loss)
        
        # 反向传播
        total_loss.backward()
        # 检查梯度
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"NaN gradient detected in {name}")
                # 处理NaN梯度，例如裁剪或跳过更新
                print(f'Epoch {epoch}, Loss: {total_loss.item():.4f}, '
                  f'Angle: {angle_loss.item():.4f}, '
                  f'Intersect: {intersect_loss.item():.4f}, '
                  f'Valley: {slope_loss.item():.4f}, '
                  f'Smooth: {smooth_loss.item():.4f}, '
                  f'Deform: {deform_loss.item():.4f}')
                1/0
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {total_loss.item():.4f}, '
                  f'Angle: {angle_loss.item():.4f}, '
                  f'Intersect: {intersect_loss.item():.4f}, '
                  f'Valley: {slope_loss.item():.4f}, '
                  f'Smooth: {smooth_loss.item():.4f}, '
                  f'Deform: {deform_loss.item():.4f}')
    
    # 返回优化后的顶点
    with torch.no_grad():
        optimized_vertices = model(original_vertices)
    
    return optimized_vertices.cpu().numpy()
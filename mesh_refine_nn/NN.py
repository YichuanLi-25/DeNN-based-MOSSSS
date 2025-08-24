import torch
import torch.nn as nn
import torch.optim as optim
from Lossfunction import compute_face_normals,angle_constraint_loss,total_variation_loss
from ValleyEdgeDetector import EdgeFacerIntersectionDetector



class MeshOptimizer(nn.Module):
    """
    三维网格优化神经网络
    
    输入: 原始网格顶点坐标 V [N, 3]
    输出: 优化后的网格顶点坐标 V' [N, 3]
    
    目标:
    1. 所有面法向量与水平面夹角 > A (硬约束)
    2. 无自交面 (硬约束)
    3. 总坐标变化最小化
    """
    def __init__(self, input_size, hidden_size=256):
        super(MeshOptimizer, self).__init__()
        # 定义网络结构 - 简单的MLP
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
  
        self.fc3 = nn.Linear(hidden_size, 3)  # 输出每个点的3D坐标变化量
        
        # 激活函数
        self.leaky_relu = nn.LeakyReLU(0.01)
        
    def forward(self, x):
        # x: [batch_size, 3] 原始顶点坐标
        h = self.leaky_relu(self.fc1(x))
        h = self.leaky_relu(self.fc2(h))
      
        delta = self.fc3(h)  # 坐标变化量
        return x + delta  # 返回新坐标


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
    model = MeshOptimizer(input_size=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 损失权重
    w_angle = 3.0  # 角度约束权重
    w_intersect = 30  # 自交约束权重
    w_valley = 5
    w_smooth = 0.01  # 平滑约束权重
    w_deform = 10.0  # 形变最小化权重
    w_shape = 20
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
        shape_deform_loss = detector.compute_norm_deforming_loss()

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
                     w_valley * slope_loss+
                     w_shape * shape_deform_loss)
        
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
                  f'Deform: {deform_loss.item():.4f},'
                  f'norm: {shape_deform_loss.item():.4f},')
                1/0
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {total_loss.item():.4f}, '
                  f'Angle: {angle_loss.item():.4f}, '
                  f'Intersect: {intersect_loss.item():.4f}, '
                  f'Valley: {slope_loss.item():.4f}, '
                  f'Smooth: {smooth_loss.item():.4f}, '
                  f'Deform: {deform_loss.item():.4f},'
                  f'norm: {shape_deform_loss.item():.4f},')
    
    # 返回优化后的顶点
    with torch.no_grad():
        optimized_vertices = model(original_vertices)
    
    return optimized_vertices.cpu().numpy()
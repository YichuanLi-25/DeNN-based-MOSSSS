import torch
import torch.nn as nn
import torch.optim as optim

def compute_face_normals(vertices, faces):
    """
    计算所有面的法向量
    
    参数:
    vertices: [N, 3] 顶点坐标
    faces: [M, 3] 面索引
    
    返回:
    normals: [M, 3] 每个面的单位法向量
    """
    # 获取每个面的三个顶点
    v0 = vertices[faces[:, 0]]  # [M, 3]
    v1 = vertices[faces[:, 1]]  # [M, 3]
    v2 = vertices[faces[:, 2]]  # [M, 3]
    
    # 计算两个边向量
    edge1 = v1 - v0  # [M, 3]
    edge2 = v2 - v0  # [M, 3]
    
    # 叉积得到法向量
    normals = torch.cross(edge1, edge2)  # [M, 3]
    
    # 归一化
    normals = normals / (torch.norm(normals, dim=1, keepdim=True) )
    
    return normals

def angle_constraint_loss(normals, min_angle_deg):
    """
    计算法向量角度约束损失
    
    参数:
    normals: [M, 3] 面法向量
    min_angle_deg: 最小允许角度(度)
    
    返回:
    loss: 违反角度约束的惩罚项
    """
    # 水平面法向量 (z轴)
    horizontal_normal = torch.tensor([0, 0, 1.0], dtype=torch.float32).to(normals.device)
    
    # 计算每个法向量与水平面的夹角 [M]
    cos_angles = (torch.sum(normals * horizontal_normal, dim=1))  # 点积
    angles = torch.acos(torch.clamp(cos_angles, -1.0, 1.0))  # 反余弦得到弧度

    # 转换为角度
    angles_deg = torch.abs(torch.rad2deg(angles))
    # print("angles_deg_length:",len(angles_deg),min_angle_deg)
    # for i in angles_deg.tolist():
    #     if i < 70:
    #         print(i)

    # 1/0
    
    # 计算违反约束的惩罚 (小于min_angle的部分)
    violation = torch.relu(min_angle_deg - angles_deg)  # ReLU激活
    # print(violation)
    # for i in violation.tolist():
    #     if i > 0:
    #         print("violation:",i)
    #1/0
    # 返回平均违反程度
    return torch.mean(violation)

def self_intersection_penalty(vertices, faces):
    """
    估算自交惩罚项 (简化版)
    
    注意: 精确的自交检测计算成本很高，这里使用简化近似
    
    参数:
    vertices: [N, 3] 顶点坐标
    faces: [M, 3] 面索引
    
    返回:
    penalty: 自交惩罚项
    """
    # 这里使用一个简化方法 - 检测是否有面之间的距离过近
    # 实际应用中可能需要更复杂的自交检测算法
    
    # 计算每个面的中心点 [M, 3]
    face_centers = (vertices[faces[:, 0]] + vertices[faces[:, 1]] + vertices[faces[:, 2]]) / 3.0
    
    # 计算所有面中心点之间的距离矩阵 [M, M]
    dist_matrix = torch.cdist(face_centers, face_centers)
    
    # 将对角线设为无穷大 (避免与自身比较)
    mask = torch.eye(dist_matrix.size(0)).bool().to(vertices.device)
    dist_matrix = dist_matrix.masked_fill(mask, float('inf'))
    
    # 找出过近的面对
    min_dist = 0.01  # 阈值
    close_pairs = (dist_matrix < min_dist).float()
    
    # 惩罚项与过近距离成反比
    penalty = torch.sum(1.0 / (dist_matrix + 1e-8) * close_pairs)
    
    return penalty

def total_variation_loss(vertices, faces):
    """
    计算总变分损失，鼓励网格平滑
    
    参数:
    vertices: [N, 3] 顶点坐标
    faces: [M, 3] 面索引
    
    返回:
    loss: 总变分损失
    """
    # 获取每个面的三个顶点
    v0 = vertices[faces[:, 0]]  # [M, 3]
    v1 = vertices[faces[:, 1]]  # [M, 3]
    v2 = vertices[faces[:, 2]]  # [M, 3]
    
    # 计算三个边的长度变化
    edge1 = torch.norm(v1 - v0, dim=1)
    edge2 = torch.norm(v2 - v1, dim=1)
    edge3 = torch.norm(v0 - v2, dim=1)
    
    # 鼓励边长变化小
    loss = torch.mean(edge1**2 + edge2**2 + edge3**2)
    
    return loss

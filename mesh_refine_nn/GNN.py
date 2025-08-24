import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ValleyEdgeDetector import ValleyEdgeDetector
from Lossfunction import compute_face_normals,angle_constraint_loss,total_variation_loss
def train_mesh_optimization(original_vertices, faces, min_angle_deg=30, num_epochs=1000):
    """
    训练网格优化模型（使用图神经网络）
    
    参数:
    original_vertices: [N, 3] 原始顶点坐标
    faces: [M, 3] 面索引
    min_angle_deg: 最小允许角度(度)
    num_epochs: 训练轮数
    
    返回:
    optimized_vertices: 优化后的顶点坐标
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 转换为PyTorch张量
    original_vertices = torch.tensor(original_vertices, dtype=torch.float32).to(device)
    faces = torch.tensor(faces, dtype=torch.long).to(device)
    
    # 从面数据构建边索引 [2, num_edges]
    edge_index = build_edge_index(faces)
    edge_index = edge_index.to(device)
    
    # 初始化图神经网络模型
    num_nodes = original_vertices.size(0)
    model = MeshGNN(num_nodes, edge_index).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 损失权重
    w_angle = 5.0    # 角度约束权重
    w_valley = 5.0    # 山谷约束权重
    w_smooth = 0.01   # 平滑约束权重
    w_deform = 10.0   # 形变最小化权重
    
    detector = ValleyEdgeDetector(faces)
    
    # 训练循环
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # 前向传播 - 获取优化后的顶点
        optimized_vertices = model(original_vertices)
        
        # 计算各项损失
        # 1. 法向量角度约束
        normals = compute_face_normals(optimized_vertices, faces)
        angle_loss = angle_constraint_loss(normals, min_angle_deg)
        
        # 2. 平滑约束
        smooth_loss = total_variation_loss(optimized_vertices, faces)
        
        # 3. 形变最小化
        deform_loss = torch.mean(torch.norm(optimized_vertices - original_vertices, dim=1))

        # 4. 山谷角度损失
        slope_loss, angles = detector.compute_slope_angle_loss_all_edges(
            V=optimized_vertices, 
            A_deg=min_angle_deg
        )
        
        # 总损失
        total_loss = (w_angle * angle_loss + 
                     w_smooth * smooth_loss + 
                     w_deform * deform_loss +
                     w_valley * slope_loss)
        
        # 反向传播
        total_loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {total_loss.item():.4f}, '
                  f'Angle: {angle_loss.item():.4f}, '
                  f'Valley: {slope_loss.item():.4f}, '
                  f'Smooth: {slope_loss.item():.4f}, '
                  f'Deform: {deform_loss.item():.4f}')
    
    # 返回优化后的顶点
    with torch.no_grad():
        optimized_vertices = model(original_vertices)
    
    return optimized_vertices.cpu().numpy()

def build_edge_index(faces):
    """从面数据构建边索引 [2, num_edges]"""
    edges = set()
    
    # 遍历所有面，提取边
    for face in faces:
        # 三角形的三条边
        edges.add((face[0].item(), face[1].item()))
        edges.add((face[1].item(), face[2].item()))
        edges.add((face[2].item(), face[0].item()))
    
    # 转换为双向边
    bidirectional_edges = []
    for u, v in edges:
        bidirectional_edges.append([u, v])
        bidirectional_edges.append([v, u])  # 添加反向边
    
    # 创建边索引张量
    edge_index = torch.tensor(bidirectional_edges, dtype=torch.long).t().contiguous()
    return edge_index

class MeshGNN(nn.Module):
    """图神经网络网格优化模型"""
    def __init__(self, num_nodes, edge_index, input_dim=3, hidden_dim=64, output_dim=3, num_layers=3):
        super().__init__()
        self.num_nodes = num_nodes
        self.edge_index = edge_index
        self.num_layers = num_layers
        
        # 构建邻接矩阵（包括自环）
        self.adj = self.build_adjacency_matrix(edge_index, num_nodes)
        
        # GNN层
        self.convs = nn.ModuleList()
        self.convs.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(nn.Linear(hidden_dim, hidden_dim))
        self.convs.append(nn.Linear(hidden_dim, output_dim))
        
        # 残差连接
        self.residual = nn.Linear(input_dim, output_dim)
        
    def build_adjacency_matrix(self, edge_index, num_nodes):
        """构建归一化的邻接矩阵（稀疏张量）"""
        # 添加自环
        self_loops = torch.arange(0, num_nodes, device=edge_index.device).repeat(2, 1)
        edge_index_with_loops = torch.cat([edge_index, self_loops], dim=1)
        
        # 计算度矩阵
        row, col = edge_index_with_loops
        deg = torch.zeros(num_nodes, device=edge_index.device)
        deg = deg.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))
        
        # 归一化因子 (D^{-1/2})
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        # 计算归一化边权重
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # 创建稀疏邻接矩阵
        return torch.sparse_coo_tensor(
            edge_index_with_loops, 
            norm, 
            (num_nodes, num_nodes)
        )
    
    def forward(self, x):
        """前向传播"""
        identity = x
        
        # 图卷积操作
        for i, conv in enumerate(self.convs):
            # 消息传播 (AX)
            x = torch.sparse.mm(self.adj, x)
            # 线性变换
            x = conv(x)
            # 非线性激活（最后一层除外）
            if i < self.num_layers - 1:
                x = F.relu(x)
        
        # 残差连接
        residual = self.residual(identity)
        return x + residual

# 注意：以下函数需要根据你的具体实现进行适配
# compute_face_normals, angle_constraint_loss, 
# total_variation_loss, ValleyEdgeDetector 等
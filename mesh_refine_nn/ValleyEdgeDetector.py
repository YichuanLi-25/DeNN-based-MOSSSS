import torch
import torch.nn as nn
import numpy as np

class ValleyEdgeDetector(nn.Module):
    def __init__(self, F):
        """
        F: 三角网格的面索引 (m, 3) 的 CPU 张量
        """
        super().__init__()
        self.register_buffer('int_edge_vertices', None)
        self.register_buffer('edge_tri_indices', None)
        self.register_buffer('_F', F.clone().detach())  # 保存面索引
        self._precompute_topology(F)
        self.face_normals = None
    
    def _precompute_topology(self, F):
        """预处理拓扑信息（仅在初始化时运行一次）"""
        device = F.device
        
        # 构建边到三角形的映射
        edge_dict = {}
        for i, tri in enumerate(F.cpu().numpy()):
            v0, v1, v2 = sorted(tri)
            edges = [(min(v0, v1), max(v0, v1)),
                     (min(v1, v2), max(v1, v2)),
                     (min(v0, v2), max(v0, v2))]
            for e in edges:
                edge_dict.setdefault(e, []).append(i)
        
        # 提取内部边（被两个三角形共享）
        int_edges = []
        edge_tri_indices = []
        edge_third_vertices = []
        
        for edge, tri_list in edge_dict.items():
            if len(tri_list) == 2:  # 内部边
                t1, t2 = tri_list
                v0, v1 = edge
                
                # 获取两个三角形的第三个顶点
                tri1_verts = set(F[t1].tolist())
                third1 = (tri1_verts - {v0, v1}).pop()
                
                tri2_verts = set(F[t2].tolist())
                third2 = (tri2_verts - {v0, v1}).pop()
                
                int_edges.append(edge)
                edge_tri_indices.append([t1, t2])
                edge_third_vertices.append([third1, third2])
        
        # 转换为张量 (E_int, 2)
        int_edges_tensor = torch.tensor(int_edges, dtype=torch.long, device=device)
        edge_tri_indices_tensor = torch.tensor(edge_tri_indices, dtype=torch.long, device=device)
        edge_third_vertices_tensor = torch.tensor(edge_third_vertices, dtype=torch.long, device=device)
        
        # 组合为 (E_int, 4) 的张量: [v0, v1, third1, third2]
        self.int_edge_vertices = torch.cat([
            int_edges_tensor, 
            edge_third_vertices_tensor
        ], dim=1)
        
        self.edge_tri_indices = edge_tri_indices_tensor
    
    def compute_valley_edges(self, V, height_channel=2):
        """
        计算山谷边
        
        参数:
            V: 顶点坐标 (n, 3) 的 GPU 张量
            height_channel: 高度所在的通道 (默认为 z 轴)
        
        返回:
            valley_flag: 布尔张量 (E_int,)，True 表示山谷边
            dot1, dot2: 点积值 (用于调试)
        """
        # 步骤1: 计算所有三角形的梯度
        s_tris = self.compute_triangle_gradients(V, height_channel)
        
        # 步骤2: 计算内部边的山谷属性
        return self.compute_edge_properties(V, s_tris, height_channel)
    
    def compute_triangle_gradients(self, V, height_channel):
        """计算所有三角形的流方向 (向量化实现)"""
        # 提取三角形顶点 (m, 3, 3)
        tris = V[self.F]
        
        # 设置参考点 v0
        v0 = tris[:, 0, :]
        v1 = tris[:, 1, :]
        v2 = tris[:, 2, :]
        
        # 计算边向量
        U = v1 - v0
        V_vec = v2 - v0
        
        # 计算高度差 (使用指定通道)
        z0 = v0[:, height_channel]
        dz1 = v1[:, height_channel] - z0
        dz2 = v2[:, height_channel] - z0
        
        # 计算点积 (向量化)
        dot_uu = (U * U).sum(dim=1)
        dot_uv = (U * V_vec).sum(dim=1)
        dot_vv = (V_vec * V_vec).sum(dim=1)
        
        # 构造线性系统
        A = torch.stack([dot_uu, dot_uv, dot_uv, dot_vv], dim=1).view(-1, 2, 2)
        B = torch.stack([dz1, dz2], dim=1).unsqueeze(-1)
        
        # 解线性方程组 (添加小正则项避免数值不稳定)
        try:
            X = torch.linalg.solve(A, B)
        except:
            I = torch.eye(2, device=A.device).unsqueeze(0)
            X = torch.linalg.solve(A + 1e-6 * I, B)
        
        # 提取系数
        a = X[:, 0, 0]
        b = X[:, 1, 0]
        
        # 计算梯度 (m, 3)
        grad = a.unsqueeze(1) * U + b.unsqueeze(1) * V_vec
        
        # 返回流方向 (负梯度)
        return -grad
    
    def compute_edge_properties(self, V, s_tris, height_channel):
        """计算内部边的山谷属性"""
        # 提取边的顶点索引
        idx = self.int_edge_vertices  # (E_int, 4)
        
        # 获取顶点坐标
        A = V[idx[:, 0]]
        B = V[idx[:, 1]]
        C = V[idx[:, 2]]
        D = V[idx[:, 3]]
        
        # 计算公共边向量
        AB = B - A
        
        # 计算三角形1的内向法向量
        AC = C - A
        proj_factor1 = (AC * AB).sum(dim=1, keepdim=True) / (AB * AB).sum(dim=1, keepdim=True).clamp(min=1e-8)
        h1 = AC - proj_factor1 * AB
        
        # 计算三角形2的内向法向量
        AD = D - A
        proj_factor2 = (AD * AB).sum(dim=1, keepdim=True) / (AB * AB).sum(dim=1, keepdim=True).clamp(min=1e-8)
        h2 = AD - proj_factor2 * AB
        
        # 获取相邻三角形的流方向
        t1_idx = self.edge_tri_indices[:, 0]
        t2_idx = self.edge_tri_indices[:, 1]
        s1 = s_tris[t1_idx]
        s2 = s_tris[t2_idx]
        
        # 计算点积
        dot1 = (s1 * h1).sum(dim=1)
        dot2 = (s2 * h2).sum(dim=1)
        
        # 判断山谷边
        valley_flag = (dot1 > 0) & (dot2 > 0)
        
        return valley_flag, dot1, dot2

    def compute_slope_angle_loss(self, V, valley_flag, A, height_channel=2):
        """
        计算山谷边的坡度角度损失
        
        参数:
            V: 顶点坐标 (n, 3) 的 GPU 张量
            valley_flag: 布尔张量 (E_int,)，表示哪些内部边是山谷边
            A: 坡度角度阈值 (度)，小于此值的边将被惩罚
            height_channel: 高度所在的通道 (默认为 z 轴)
        
        返回:
            loss: 坡度角度损失值
            angles: 所有山谷边的坡度角度 (度)
        """
        # 如果没有山谷边，直接返回0损失
        if not valley_flag.any():
            return torch.tensor(0.0, device=V.device), None
        
        # 提取山谷边的顶点索引
        valley_edges = self.int_edge_vertices[valley_flag]  # (E_valley, 4)
        
        # 获取山谷边的两个端点
        v0 = V[valley_edges[:, 0]]
        v1 = V[valley_edges[:, 1]]
        
        # 计算边向量
        edge_vec = v1 - v0
        
        # 计算水平距离和高度差
        dx = edge_vec[:, 0]
        dy = edge_vec[:, 1]
        dz = edge_vec[:, height_channel]
        
        # 计算水平距离 (忽略高度分量)
        horizontal_dist = torch.sqrt(dx**2 + dy**2 + 1e-8)
        
        # 计算坡度角度 (弧度)
        slope_angles_rad = torch.atan2(torch.abs(dz), horizontal_dist)
        
        # 转换为度
        slope_angles_deg = torch.rad2deg(slope_angles_rad)
        
        # 将角度阈值A转换为张量
        A_tensor = torch.tensor(A, dtype=torch.float32, device=V.device)
        
        # 计算损失：只惩罚角度小于A的山谷边，大于A的取0
        angle_diff = A_tensor - slope_angles_deg
        # 使用 ReLU 确保只惩罚小于阈值的情况
        losses = torch.nn.functional.relu(angle_diff) ** 2
        #print(len(losses))
        # 计算所有山谷边的平均损失（包括损失为0的边）
        loss = losses.mean()
        
        return loss, slope_angles_deg
    def compute_edge_normals(self):
        """
        计算所有内部边的平均法向量
        
        参数:
            V: 顶点坐标 (n, 3) 的张量
        
        返回:
            edge_normals: 边的平均法向量 (E_int, 3)
        """
        # 计算所有三角形的法向量
        tri_normals = self.face_normals
        # 获取每条边的两个相邻三角形的法向量
        tri1_idx = self.edge_tri_indices[:, 0]
        tri2_idx = self.edge_tri_indices[:, 1]
        normals1 = tri_normals[tri1_idx]
        normals2 = tri_normals[tri2_idx]
        
        # 计算平均法向量（相加后归一化）
        avg_normals = normals1 + normals2
        norm = torch.norm(avg_normals, dim=1, keepdim=True).clamp(min=1e-18)
        return avg_normals / norm
    

            

    
    def compute_slope_angle_loss(self, V,  A_deg, height_channel=2):
        """
        基于边法向与竖直方向夹角的坡度损失
        
        参数:
            V: 顶点坐标 (n, 3) 的张量
            valley_flag: 布尔张量 (E_int,)，表示哪些边是山谷边
            A_deg: 坡度角度阈值 (度)，法向夹角小于此值的边将被惩罚
            height_channel: 高度所在的通道 (默认为 z 轴)，用于竖直方向
        
        返回:
            loss: 坡度角度损失值
            angles: 所有山谷边的法向夹角 (度)
        """

        
        # 计算所有内部边的平均法向量
        edge_normals = self.compute_edge_normals(V)
        

        
        # 计算法向量与竖直方向(0,0,1)的夹角
        # 使用点积：cosθ = |n · up|，其中up = [0,0,1]
        up_vector = torch.zeros_like(edge_normals)
        up_vector[:, height_channel] = 1.0  # 竖直方向
        
        # 计算点积 (绝对值确保夹角在0-90度之间)
        dot_products = torch.abs(torch.sum(edge_normals * up_vector, dim=1))
        
        # 计算夹角（弧度）
        angles_rad = torch.acos(torch.clamp(dot_products, -1.0, 1.0))
        
        # 转换为度
        angles_deg = torch.rad2deg(angles_rad)
        
        # 将角度阈值A转换为张量
        A_rad = torch.deg2rad(torch.tensor(90-A_deg, device=V.device))
        cosA = torch.cos(A_rad)  # 阈值的余弦值
        
        # 计算损失：夹角小于A_deg时进行惩罚
        # 损失 = max(0, cosA - |n·up|)^2
        # 当|n·up| > cosA 时（即夹角小于A），产生损失
        diff = cosA - dot_products
        losses = torch.nn.functional.relu(diff) ** 2
        
        # 计算所有山谷边的平均损失
        loss = losses.mean()
        
        return loss, angles_deg
    def compute_triangle_normals(self,V):
        """
        计算所有三角形的法向量（归一化）
        
        参数:
            V: 顶点坐标 (n, 3) 的张量
        
        返回:
            normals: 三角形法向量 (m, 3)
        """
        # 提取三角形顶点
        tris = V[self.F]
        
        # 计算边向量
        v0 = tris[:, 0, :]
        v1 = tris[:, 1, :]
        v2 = tris[:, 2, :]
        
        # 计算法向量 (叉积)
        edge1 = v1 - v0
        edge2 = v2 - v0
        normals = torch.cross(edge1, edge2, dim=1)
        
        # 归一化
        norm = torch.norm(normals, dim=1, keepdim=True).clamp(min=1e-8)
        self.face_normals = normals / norm
    def compute_slope_angle_loss_all_edges(self, A_deg, height_channel=2):
        """
        计算所有内部边的坡度损失（不限于山谷边）
        
        参数:
            V: 顶点坐标 (n, 3) 的张量
            A_deg: 坡度角度阈值 (度)
            height_channel: 高度所在的通道 (默认为 z 轴)
        
        返回:
            loss: 坡度角度损失值
            angles: 所有内部边的法向夹角 (度)
        """
        # 计算所有内部边的平均法向量
        
        self.edge_normals = self.compute_edge_normals()
        
        
        # 计算法向量与竖直方向(0,0,1)的夹角
        up_vector = torch.zeros_like(self.edge_normals)
        up_vector[:, height_channel] = 1.0  # 竖直方向
        
        # 计算点积 (绝对值确保夹角在0-90度之间)
        dot_products = torch.abs(torch.sum(self.edge_normals * up_vector, dim=1))
        
        # 计算夹角（弧度）
        angles_rad = torch.acos(torch.clamp(dot_products, -1.0, 1.0))
        
        # 转换为度
        angles_deg = torch.abs(torch.rad2deg(angles_rad))
        #print("angles_deg:",angles_deg)
        
        #angles_deg = torch.abs(90-angles_deg)
        #print(angles_deg)
        
        # 将角度阈值A转换为张量
        A_deg = (torch.tensor(A_deg, device=self.face_normals.device))
        #print(A_rad,A_deg)
        #1/0
        # 计算损失：夹角小于A_deg时进行惩罚
        diff =  A_deg - angles_deg
        #print("diff:",diff)
        losses = torch.nn.functional.relu(diff) 
        #losses = losses ** 0.5
        #print('loss:',losses)
        
        # 计算所有边的平均损失
        loss = losses.mean()
        #print('mean',loss)
        #1/0
        return loss, angles_deg
    
    # def compute_slope_angle_loss_all_edges(self, V, A_deg , recomput_normals = False):
    #     """
    #     计算所有内部边的夹角损失（不限于山谷边）
        
    #     参数:
    #         V: 顶点坐标 (n, 3) 的张量
    #         A_deg: 相邻面角度阈值 (度)
           
        
    #     返回:
    #         loss: 角度损失值
    #         angles: 所有内部边的法向夹角 (度)
    #     """
    #     # 计算所有内部边的平均法向量
    #     if self.face_normals is None or recomput_normals:
    #         edge_normals = self.compute_edge_normals(V)
    #     else:
    #         edge_normals = self.edge_normals
    #         self.edge_normals = edge_normals

        
        
        # 计算点积 (绝对值确保夹角在0-90度之间)
        dot_products = torch.abs(torch.sum(edge_normals * up_vector, dim=1))
        
        # 计算夹角（弧度）
        angles_rad = torch.acos(torch.clamp(dot_products, -1.0, 1.0))
        
        # 转换为度
        angles_deg = torch.abs(torch.rad2deg(angles_rad))
        #print("angles_deg:",angles_deg)
        
        #angles_deg = torch.abs(90-angles_deg)
        #print(angles_deg)
        
        # 将角度阈值A转换为张量
        A_deg = (torch.tensor(A_deg, device=V.device))
        #print(A_rad,A_deg)
        #1/0
        # 计算损失：夹角小于A_deg时进行惩罚
        diff =  A_deg - angles_deg
        #print("diff:",diff)
        losses = torch.nn.functional.relu(diff) 
        #losses = losses ** 0.5
        #print('loss:',losses)
        
        # 计算所有边的平均损失
        loss = losses.mean()
        #print('mean',loss)
        #1/0
        return loss, angles_deg

    @property
    def F(self):
        """访问面索引 (自动处理设备)"""
        return self._F.to(self.int_edge_vertices.device) if self.int_edge_vertices is not None else self._F

class EdgeFacerIntersectionDetector(ValleyEdgeDetector):
    def __init__(self, F , V):
        super().__init__(F)
        self.compute_triangle_normals(V)

        o_cos = self.compute_edge_angels()
        self.o_cos = o_cos.detach().clone()
        self.o_norm = self.face_normals.detach().clone()
    
    def compute_norm_deforming_loss(self):
        losses = self.face_normals - self.o_norm
        losses = torch.norm(losses,dim=1)
        return losses.mean()

    
    def compute_intersection_loss(self):
        coss = self.compute_edge_angels()
        #print(angles,self.o_angles)
        d_coss = coss - self.o_cos
        angles = torch.relu(d_coss)
        #print(angles)
        angle_loss = angles.mean()
        #print(angle_loss)
        if  torch.isnan(angles).any():
            raise Exception('nan')
        return angle_loss
    def compute_edge_angels(self):
        """
        计算所有内部边二面角
        
        参数:
            V: 顶点坐标 (n, 3) 的张量
        
        返回:
            edge_angles: 边的平均法向量与竖直方向的夹角 (度)
        """
        # 获取所有内部边的平均法向量
        tri_normals = self.face_normals
        # 获取每条边的两个相邻三角形的法向量
        tri1_idx = self.edge_tri_indices[:, 0]
        tri2_idx = self.edge_tri_indices[:, 1]
        A = tri_normals[tri1_idx]
        B = tri_normals[tri2_idx] 
        # 1. 计算点积（沿向量维度求和）
        dot_products = (A * B).sum(dim=1)

        # 2. 计算模长
        norm_A = torch.norm(A, dim=1)
        norm_B = torch.norm(B, dim=1)  
        if (norm_A ==0).any():
            print (norm_A)
        if (norm_B ==0).any():
            print(norm_B)
        # 3. 计算余弦值
        cosines = dot_products / (norm_A * norm_B+1e-8)

        # 4. 处理数值误差（确保余弦值在[-1, 1]范围内）
        cosines = torch.clamp(cosines, -1.0, 1.0)

        # 5. 计算夹角（弧度）
        #angles_rad = torch.acos(cosines)

        return 1-cosines
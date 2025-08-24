from compas.datastructures import Mesh
# from compas.geometry import centroid_points
# #from gradient_evaluation_dart import get_vertex_gradient_from_z
# #from mesh_cutting import print_list_with_details
# #from distances import save_nested_list
# from sympy import symbols, Eq, solve
# import copy
# import numpy as np
# import math
# import time
import os
import compas_slicer.utilities as utils
from compas.files import OBJWriter 

# from gradient_descent import kill_local_critical,kill_local_criticals
# from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ValleyEdgeDetector import ValleyEdgeDetector
from NN import train_mesh_optimization
#from GNN import train_mesh_optimization

#from siren import train_mesh_optimization

# 示例使用
if __name__ == "__main__":
   
  
    input_folder_name='branches'#'wallwithholes''beam_testprint''MNNaCl1''Jul_ai''whole''beam1B''csch2''example_jun_bg''data_Y_shape' 'data_vase''data_costa_surface''data_Y_shape_o''data_vase_o''data_costa_surface_o''Jun_ab_testmultipipe'
    #'Jun_ah_testb''Jul_h''Jul_I''Jul_ab''Jul_ah''Jul_ba''table_1''Aug_ac_ex''Aug_bg''Aug_bh''example_jun_bg''table_2'
    DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), input_folder_name)
    
    OUTPUT_PATH = utils.get_output_directory(DATA_PATH)
    mesh = Mesh.from_obj(os.path.join(DATA_PATH,"mesh.obj"))
    # # 创建简单网格 (这里用随机点简化，实际应从真实网格文件读取)
    # np.random.seed(42)
    #num_vertices = 1000  # 测试用较少的点
    original_vertices,faces = mesh.to_vertices_and_faces()
    original_vertices,faces = np.array(original_vertices),np.array(faces)
    mean_vertices = np.mean(original_vertices, axis=0)
    print('mean_vertices',mean_vertices)
    original_vertices = original_vertices - mean_vertices
    scaling_factor = np.max(np.abs(original_vertices.flatten()))
    original_vertices = original_vertices / scaling_factor
    
    # 使用Delaunay三角化创建面 (仅用于演示，实际应从真实网格读取)
    # 注意: 这里仅适用于凸包，实际应用应从文件读取面信息
    #tri = Delaunay(original_vertices[:, :2])  # 仅使用x,y坐标进行2D三角化
    #faces = tri.simplices
    
    # 确保面索引在有效范围内
    #faces = np.clip(faces, 0, num_vertices-1)
    
    print(f"Original mesh: {original_vertices.shape[0]} vertices, {faces.shape[0]} faces")
    
    # 运行优化
    min_angle = 40  # 最小角度45度
    optimized_vertices = train_mesh_optimization(
        original_vertices, 
        faces, 
        min_angle_deg=min_angle,
        num_epochs=10000
    )
    
    # # 可视化结果
    # fig = plt.figure(figsize=(12, 6))
    
    # # 原始网格
    # ax1 = fig.add_subplot(121, projection='3d')
    # ax1.plot_trisurf(original_vertices[:, 0], original_vertices[:, 1], original_vertices[:, 2], 
    #                 triangles=faces, alpha=0.6)
    # ax1.set_title('Original Mesh')
    
    # # 优化后网格
    # ax2 = fig.add_subplot(122, projection='3d')
    # ax2.plot_trisurf(optimized_vertices[:, 0], optimized_vertices[:, 1], optimized_vertices[:, 2],
    #                 triangles=faces, alpha=0.6)
    # ax2.set_title('Optimized Mesh')
    
    # plt.tight_layout()
    # plt.show()
    mesh = mesh.from_vertices_and_faces(optimized_vertices * scaling_factor + mean_vertices, faces)
    obj_writer = OBJWriter(filepath= os.path.join(OUTPUT_PATH, "edited_mesh.obj"), meshes=[mesh])
    obj_writer.write()
 
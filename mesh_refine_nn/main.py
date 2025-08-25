from compas.datastructures import Mesh
import os
import compas_slicer.utilities as utils
from compas.files import OBJWriter 
import numpy as np
from NN import train_mesh_optimization
#from GNN import train_mesh_optimization
#from siren import train_mesh_optimization

# 示例使用
if __name__ == "__main__":
   
  
    input_folder_name='branches'#'wallwithholes'
    support_angle = 30   # default = 30°
    DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), input_folder_name)
    
    OUTPUT_PATH = utils.get_output_directory(DATA_PATH)
    mesh = Mesh.from_obj(os.path.join(DATA_PATH,"mesh.obj"))

    original_vertices,faces = mesh.to_vertices_and_faces()
    original_vertices,faces = np.array(original_vertices),np.array(faces)
    mean_vertices = np.mean(original_vertices, axis=0)
    original_vertices = original_vertices - mean_vertices
    scaling_factor = np.max(np.abs(original_vertices.flatten()))
    original_vertices = original_vertices / scaling_factor
    print(f"Original mesh: {original_vertices.shape[0]} vertices, {faces.shape[0]} faces")
    
    # run optimization
    
    optimized_vertices = train_mesh_optimization(
        original_vertices, 
        faces, 
        min_angle_deg=support_angle,
        num_epochs=10000
    )
    mesh = mesh.from_vertices_and_faces(optimized_vertices * scaling_factor + mean_vertices, faces)
    obj_writer = OBJWriter(filepath= os.path.join(OUTPUT_PATH, "edited_mesh.obj"), meshes=[mesh])
    obj_writer.write()
 
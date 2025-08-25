# DeNN-based-MoSSSS

## 📖 Introduction  
The goal of this project is to perform **morphological optimization with neural networks** so that input shell structures become **self-supporting for 3D printing self-supporting single shell**, eliminating the need for additional support structures.  

---

## 🖼️ Results
The figure below demonstrates three representative examples:  

- **Left: Before optimization (non-self-supporting)**  
- **Right: After optimization (self-supporting through shape adjustment)**  

<p align="center">
  <img src="results.jpg".\ width="800">
</p>

👉 From up to down, the examples correspond to: **branches, circles, wallwitholes**.

---
## Environment Setup

Create a new conda environment and note the python version.
```
conda create -n slicer_3dprint_env python==3.10
```

Check your CUDA versions, then install a pytorch version on https://pytorch.org/get-started/previous-versions/. Here is my command for my PC with CUDA 12.4.
```
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

You can validate your pytorch installation with the following command. If your pytorch version is not compatible with your CUDA version (as indicated on the PyTorch website), the output will be False.
```
python -c "import torch; print(torch.cuda.is_available())"
```

Then install compas packages for reading 3D models. They could only be installed by conda-forge commands.
```
conda install -c conda-forge compas==1.17.4 compas_slicer==0.6.1
```

Install some other packages you need for your codes.
```
pip install scipy==1.10.0
```
---

## ⚡ Quick Start Example
This project provides three demo mesh cases for direct testing.

1. Ensure that the following demo folders exist in the project root:  
   ```
   branches/mesh.obj
   circles/mesh.obj
   wallwitholes/mesh.obj
   ```
2. Open `mesh_refine_nn/main.py` and set the `foldername`, e.g.:
   ```python
   foldername = "branches"
   ```
3. Run the code:
   ```bash
   cd mesh_refine_nn
   python main.py
   ```
4. The optimized mesh will be saved in:
   ```
   branches/output/
   ```

---

## ⚙️ Custom Usage


1. **Prepare your input mesh**  
   - Create a new folder under the project root, e.g. `example_mesh/`.  
   - Place your triangular mesh inside and rename it to:  
     ```
     mesh.obj
     ```

2. **Modify `main.py` settings**  
   - Open `mesh_refine_nn/main.py`.  
   - Set the folder name, e.g.:
     ```python
     foldername = "example_mesh"
     ```  
   - You may also adjust the **self-supporting angle limit**:
     ```python
     support_angle = 30   # default = 30°
     ```

3. **Run**  
   ```bash
   cd mesh_refine_nn
   python main.py
   ```

4. **Check the results**  
   The optimized mesh will appear in:  
   ```
   <your_folder>/output/
   ```

---

## 📂 Project Structure
```
.
├── mesh_refine_nn/          # Core neural network and main function
│   └── main.py/             
├── demo_case1/              # Demo input mesh 1
│   ├── mesh.obj
│   └── output/              # Results
├── demo_case2/              # Demo input mesh 2
├── demo_case3/              # Demo input mesh 3
├── example_mesh/            # Example folder for user input
│   ├── mesh.obj
│   └── output/              
└── README.md
```

---

## 🔧 Configurable Parameters
- `foldername`: Folder name of the input mesh  
- `support_angle`: Self-supporting angle threshold (degrees, default = 45°)



# 1. Environment Setup

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

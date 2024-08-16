#! /bin/bash
pip install opencv-python matplotlib open3d torch torchvision os
pip install 'git+https://github.com/facebookresearch/segment-anything.git'

./scripts/download_models.sh
# get SAM checkpoint 
mkdir sam_ckpts
wget -P sam_ckpts https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget -P sam_ckpts https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
wget -P sam_ckpts https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth


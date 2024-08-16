# Mask annotation

This submodule is used to first use [SAM](https://segment-anything.com/) to get a mask at t=0, and then use [XMem](https://hkchengrex.com/XMem/).

## Installation

```./install.sh``

## SAM Annotation
We first need to annotate the dataset to get the masks at t=0.
For any given dataset, we can annotate the images at t=0 by pressing 5 points for each image. 
We prepared a script which takes in a NeRF json and finds the images at t=0. The results are stored in a `SAM_annotations` dir, created in the root dir of the input json. E.g.:

```python3 XMem/annotate_dataset.py --json robo360/data/batch2/xarm6_fold_tshirt_DNeRF/transforms_train.json ``

Saves results to `robo360/data/batch2/xarm6_fold_tshirt_DNeRF/masks/`.

## Running XMem

We use XMem to propagate the masks into future timesteps. Run XMem on the SAM results aquired above by running:

```python3 eval_xmem.py --generic_path robo360/data/batch2/xarm6_fold_tshirt_DNeRF/masks/ --dataset G```


The dataset is now ready for DeformGS.
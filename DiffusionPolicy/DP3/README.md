# DexH2R: Diffusion Policy (DP3) Implementation Guide

## 🛠️ Environment Setup


Create conda environment following the instruction on [Diffusion Policy 3D](https://github.com/YanjieZe/3D-Diffusion-Policy/blob/master/INSTALL.md)
```console
# Activate environment
$ conda activate dp3

# Install dependencies for our task setting
$ pip install pytransform3d pytorch_kinematics trimesh open3d pillow plotly typing-extensions
```
For [pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) and [csdf](https://github.com/wrc042/CSDF/blob/main/README.md), follow the instructions on their github repo to install.

## 📥 Dataset Preparation
1. Download dataset from [Google Drive]
2. Place all dataset files in the `dataset/DiffusionPolicy_dataset` directory
    - Put `train_val.zarr` in `train/whole` folder and put `train_1.zarr`, `train_2.zarr`, `train_3.zarr` in `train/split` folder. 
    - Put `val.zarr` in `val` folder.
    - Put `test.zarr` in `test` folder.

## 🎁 Checkpoint Download
- Download checkpoint with `horizon=8`, `n_obs_steps=5`, `n_action_steps=2` from [Google Drive].

## 🏋️ Training Configuration

### Option 1: Full Dataset Training (Recommended for high-memory systems)
```console
(dp3)$ bash scripts/train_policy.sh dp3_ours ours_fastgrasp 0618 0 0
```

### Option 2: Memory-Efficient Training (For systems with <200GB RAM)
```console
(dp3)$ bash scripts/train_small_mem_policy.sh dp3_ours ours_fastgrasp_small_mem 0618 0 0
```

> **Note**: 
> - Both methods produce equivalent results
> - The memory-efficient version uses pre-calculated normalizers
> - Modify hyperparameters in the respective YAML files

## 🔍 Evaluation

```console
(dp3)$ bash scripts/eval_policy.sh dp3_ours ours_fastgrasp 0618 0 0
```
- Also need to load the pre-calculated `normalizer`.(Have been done by default.)

### Evaluation Parameters (Configure in YAML):
- `eval_folder`: Visualization output directory
- `visualize`: Whether to save the visualization files.
- `total_infer_frame_threshold`: Number of frames before the model can interpolate.(To prevent model from grasping the static object when the trajectory begins.) 
- `interpolation_threshold`: Control parameter
- `checkpoint_path`: Model weights location


## Others:

### Dexterous Hand Model dependencies:
- `e3m5_hand_model.py`, `rot6d.py`, which are included in the `DP` folder.
---

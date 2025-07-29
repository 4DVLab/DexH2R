# Diffusion Policy

## Existing Dataset Usage

### For 2D Diffusion Policy (DP)
- **Training & Evaluation**: 
  - Uses `.zarr` format datasets as described in the paper
  - Refer to the [DP README](https://github.com/wang-youzhuo/DexH2R/blob/main/DiffusionPolicy/DP/README.md) for detailed usage instructions

### For 3D Diffusion Policy (DP3)
- **Training & Evaluation**:
  - Uses specialized `.zarr` datasets
  - See [DP3 README](https://github.com/wang-youzhuo/DexH2R/blob/main/DiffusionPolicy/DP3/README.md) for implementation details

## Custom Dataset Generation

You can generate your own `.zarr` datasets from our base datasets with flexible configuration options:

### Key Configuration Parameters
- `grasping_period`: 
  - `False` (default): Excludes static grasping periods where the object and the dexterous hand both are static (focuses on dynamic grasping only)
  - `True`: Includes both grasping and tracking periods

### Generation Scripts
1. **`merge_dataset.py`**: Combines and preprocesses raw data (excludes rgb and depth images)
2. **`merge_images.py`**: Combines and preprocesses raw rgb and depth images
2. **`merge_zarr.py`**: Chooses and converts processed data to `.zarr` format

<!-- ### Dataset Type Specifications
| Dataset Type | Required Settings          |
|--------------|---------------------------|
| DP (2D)      | `image=True`, `depth=True` |
| DP3 (3D)     | Use default settings       | -->

### Usage Example
#### 
```bash
cd DiffusionPolicy
# Merge the dataset to .npy
# - Include the static grasping period
python ./dataset_generation/merge_dataset.py --grasping_period
python ./dataset_generation/merge_images.py --grasping_period
# - Exclude the static grasping period
python ./dataset_generation/merge_dataset.py 
python ./dataset_generation/merge_images.py

# Merge all the .npy files to .zarr folder
# - For dp
python merge_zarr.py --algo dp
# - For dp3
python merge_zarr.py --algo dp3

# Or Just single run the script below:
# - For dp:
cd DiffusionPolicy
chmod +x run_dataset_generation_dp.sh
./run_dataset_generation_dp.sh
# - For dp3:
cd DiffusionPolicy
chmod +x run_dataset_generation_dp3.sh
./run_dataset_generation_dp3.sh
```









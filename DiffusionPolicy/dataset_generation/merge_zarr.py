import zarr
import numpy as np
import argparse

# create argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--algo', 
                   type=str,
                   choices=['dp', 'dp3'],  # only dp or dp3
                   required=True,          # must provide
                   help='algorithm: dp or dp3')
# parse arguments
args = parser.parse_args()
# use arguments
algo = args.algo

merge_dir = "../dataset/DexH2R_merge_dataset"

action_path = f"{merge_dir}/action.npy"
agent_pos_path = f"{merge_dir}/agent_pos.npy"
if algo == 'dp':
    rgb_image_path = f"{merge_dir}/image.npy"
    depth_image_path = f"{merge_dir}/depth.npy"
velocity_path = f"{merge_dir}/velocity.npy"
final_grasp_path = f"{merge_dir}/final_grasp.npy"
final_grasp_group_path = f"{merge_dir}/final_grasp_group.npy"
objpcd_intact_path = f"{merge_dir}/objpcd_intact.npy"
objpcd_normal_intact_path = f"{merge_dir}/objpcd_normal_intact.npy"
obs_objpcd_path = f"{merge_dir}/obs_objpcd.npy"
episode_end_path = f"{merge_dir}/episode_ends.npy"
traj_index_path = f"{merge_dir}/traj_index.npy"


actions = np.load(action_path)  # 假设shape为(N, A)
if algo == 'dp':
    rgb_images = np.load(rgb_image_path)
    depth_images = np.load(depth_image_path)
agent_pos = np.load(agent_pos_path)
velocity = np.load(velocity_path)
final_grasp = np.load(final_grasp_path)
final_grasp_group = np.load(final_grasp_group_path)
objpcd_intact = np.load(objpcd_intact_path)
objpcd_normal_intact = np.load(objpcd_normal_intact_path)
obs_objpcd = np.load(obs_objpcd_path)
traj_index = np.load(traj_index_path)

# N: total data length
N = actions.shape[0]  
# TODO: change the zarr_file_path to the relative path
zarr_file_path = f"../dataset/DexH2R_merge_dataset/DexH2R_{args.algo}_dataset.zarr"

root = zarr.open(zarr_file_path, mode='w')
# create folder structure
data_group = root.create_group('data')  # create data folder
meta_group = root.create_group('meta')  # create meta folder

small_chunk = 200
large_chunk = 200
action_store = data_group.create_dataset('action', shape=actions.shape, dtype=actions.dtype, chunks=(large_chunk,) + actions.shape[1:])
if algo == 'dp':
    rgb_image_store = data_group.create_dataset('image', shape=rgb_images.shape, dtype=rgb_images.dtype, chunks=(small_chunk,) + rgb_images.shape[1:])
    depth_image_store = data_group.create_dataset('depth', shape=depth_images.shape, dtype=depth_images.dtype, chunks=(small_chunk,) + depth_images.shape[1:])
agent_pos_store = data_group.create_dataset('agent_pos', shape=agent_pos.shape, dtype=agent_pos.dtype, chunks=(large_chunk,) + agent_pos.shape[1:])
velocity_store = data_group.create_dataset('velocity', shape=velocity.shape, dtype=velocity.dtype, chunks=(large_chunk,) + velocity.shape[1:])
final_grasp_store = data_group.create_dataset('final_grasp', shape=final_grasp.shape, dtype=final_grasp.dtype, chunks=(large_chunk,) + final_grasp.shape[1:])
final_grasp_group_store = data_group.create_dataset('final_grasp_group', shape=final_grasp_group.shape, dtype=final_grasp_group.dtype, chunks=(large_chunk,) + final_grasp_group.shape[1:])
objpcd_intact_store = data_group.create_dataset('objpcd_intact', shape=objpcd_intact.shape, dtype=objpcd_intact.dtype, chunks=(small_chunk,) + objpcd_intact.shape[1:])
objpcd_normal_intact_store = data_group.create_dataset('objpcd_normal_intact', shape=objpcd_normal_intact.shape, dtype=objpcd_normal_intact.dtype, chunks=(small_chunk,) + objpcd_normal_intact.shape[1:])
obs_objpcd_store = data_group.create_dataset('obs_objpcd', shape=obs_objpcd.shape, dtype=obs_objpcd.dtype, chunks=(small_chunk,) + obs_objpcd.shape[1:])
traj_index_store = data_group.create_dataset('traj_index', shape=traj_index.shape, dtype=traj_index.dtype, chunks=(large_chunk,) + traj_index.shape[1:])

# write data
action_store[:] = actions
if algo == 'dp':
    rgb_image_store[:] = rgb_images
    depth_image_store[:] = depth_images
agent_pos_store[:] = agent_pos
velocity_store[:] = velocity
final_grasp_store[:] = final_grasp
final_grasp_group_store[:] = final_grasp_group
objpcd_intact_store[:] = objpcd_intact
objpcd_normal_intact_store[:] = objpcd_normal_intact
obs_objpcd_store[:] = obs_objpcd
traj_index_store[:] = traj_index

episode_ends = np.load(episode_end_path)
meta_group.create_dataset('episode_ends', data=episode_ends, chunks=(300,), dtype=episode_ends.dtype)
print(f'Zarr file created at {zarr_file_path}')
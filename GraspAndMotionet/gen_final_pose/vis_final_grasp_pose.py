import os
import sys
sys.path.append(os.getcwd())
from utils.e3m5_hand_model import get_e3m5_handmodel
import torch
from os.path import join as pjoin
from tqdm import tqdm
import argparse

def arg_init():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--obj_name", required=True, help="object name."
    )
    parser.add_argument(
        "--output_dir", required=True, help="output directory."
    )
    parser.add_argument(
        "--final_pose_dir_name", default="model_final_qpose_cvae", help="final_pose_dir_name."
    )
    args = parser.parse_args()
    return args

def main():

    args = arg_init()
    obj_name = args.obj_name
    output_dir = args.output_dir
    os.makedirs(output_dir,exist_ok=True)

    model_final_qpose_dir_path = f"./../dataset/{args.final_pose_dir_name}/"
    model_final_qpose_path = pjoin(model_final_qpose_dir_path,f"{obj_name}_final_qpose.pt")
    model_final_qpose = torch.load(model_final_qpose_path).squeeze(0)
    e3m5_hand_model = get_e3m5_handmodel("cpu")

    for data_index in tqdm(torch.arange(model_final_qpose.shape[0])):
        
        one_hand_qpose = model_final_qpose[data_index: data_index + 1]
        hand_mesh = e3m5_hand_model.get_meshes_from_q(one_hand_qpose)
        hand_mesh.export(pjoin(output_dir,f"{data_index}.ply"))
        


if __name__ == "__main__":
    main()


python gen_final_pose/gen_mesh_model_final_grasp_pose.py \
            --config-name=grasp\
            hydra/job_logging=none hydra/hydra_logging=none \
            model=cvae \
            task=grasp_gen_ur \
            task.dataset.normalize_x=True \
            task.dataset.normalize_x_trans=True \
            model.norm=True





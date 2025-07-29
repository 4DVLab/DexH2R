

python test_grasp_quality/gen_grasp_poses_for_test.py \
            --config-name=grasp\
            hydra/job_logging=none hydra/hydra_logging=none \
            model=cvae \
            task=grasp_gen_ur \
            task.dataset.normalize_x=True \
            task.dataset.normalize_x_trans=True \
            model.norm=True \
            grasp_gen_res_save_dir_path=test_meshes/grasp_test_res
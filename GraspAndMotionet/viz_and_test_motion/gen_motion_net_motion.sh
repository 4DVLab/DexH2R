# EXP_NAME=$1


# CUDA_VISIBLE_DEVICES=0 
python viz_and_test_motion/motion_net_gen_motion.py \
                hydra/job_logging=none hydra/hydra_logging=none \
                use_pretrain=true \
                pretrain_model_dir_path=experiment_result/2025-07-11_20-40-38_motion_net_no_dropout_use_grasp/ckpts \
                pretrain_model_index=369 \
                task.itp_mode=5 \
                task.dataset.use_mano_filter_cvae_collision_final_pose=false\
                task.dataset.use_mano_filter_dexgraspanything_collision_final_pose=false \
                task.dataset.use_cvae_final_pose=true \
                task.dataset.use_dexgraspanything_final_pose=false \
                task.eval_task=test \
                model.use_predict_num=5
                # task.eval_task=ablation \
                # task.dataset.past_frames=5 \
                # task.dataset.future_frames=10 \
                



# Examples:
# bash scripts/train_small_mem_policy.sh dp3_ours ours_fastgrasp_small_mem 0618  0 0

DEBUG=False
save_ckpt=True

alg_name=${1} # dp3_ours
task_name=${2} # ours_fastgrasp
config_name=${alg_name} # dp3_ours
addition_info=${3} # 0512(名字)
seed=${4} # 0
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"


# gpu_id=$(bash scripts/find_gpu.sh)
gpu_id=${5} # 0
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"


if [ $DEBUG = True ]; then
    wandb_mode=offline
    # wandb_mode=online
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
else
    # wandb_mode=online
    wandb_mode=offline
    echo -e "\033[33mTrain mode\033[0m"
fi

# cd 3D-Diffusion-Policy


export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}
python train_small_mem.py --config-name=${config_name}.yaml \
                            task=${task_name} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt}



                                
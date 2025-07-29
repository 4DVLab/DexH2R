EXP_NAME=$1
# need user to define a log save folder path

CUDA_VISIBLE_DEVICES=0 python train.py \
                --config-name=motion\
                hydra/job_logging=none hydra/hydra_logging=none \
                exp_name=${EXP_NAME} \
                model=motion_net \
                task=motion_net_gen 


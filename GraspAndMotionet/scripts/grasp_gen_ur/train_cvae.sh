EXP_NAME=$1
# need user to define a log save folder path

python train.py \
        --config-name=grasp \
        hydra/job_logging=none hydra/hydra_logging=none \
        exp_name=${EXP_NAME} \
        model=cvae \
        model.norm=True \
        task.dataset.normalize_x=True \
        task.dataset.normalize_x_trans=True \
        task=grasp_gen_ur \

CKPT=$1


if [ -z ${CKPT} ]
then
    echo "No ckpt input."
    exit
fi


echo "Without optimizer guidance."
python models/vis_youzhuo_original.py hydra/job_logging=none hydra/hydra_logging=none \
            exp_dir=${CKPT} \
            model=cvae \
            task=grasp_gen_ur \
            task.dataset.normalize_x=true \
            task.dataset.normalize_x_trans=true


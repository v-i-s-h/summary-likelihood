#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --exclude=dgx[1-7]
#SBATCH --array=1-16
#SBATCH --output=logs/job-%A-%a.out
#SBATCH --error=logs/job-%A-%a.err

module purge
module load miniconda
source activate bayes

export DISABLE_PBAR=1

OUTDIR="zoo/abl-lamsl1e-4-uniform-1000steps"
MAX_STEPS=3000
METHOD="sl"


case $SLURM_ARRAY_TASK_ID in
    1) CORRUPTION=brightness ;;
    2) CORRUPTION=canny_edges ;;
    3) CORRUPTION=dotted_line ;;
    4) CORRUPTION=fog ;;
    5) CORRUPTION=glass_blur ;;
    6) CORRUPTION=identity ;;
    7) CORRUPTION=impulse_noise ;;
    8) CORRUPTION=motion_blur ;;
    9) CORRUPTION=rotate ;;
    10) CORRUPTION=scale ;;
    11) CORRUPTION=shear;;
    12) CORRUPTION=shot_noise ;;
    13) CORRUPTION=spatter ;;
    14) CORRUPTION=stripe ;;
    15) CORRUPTION=translate ;;
    16) CORRUPTION=zigzag ;;
esac


for (( i = 1; i <= 3; i++ ))
do
    for dssize in 10000 5000 2500 1000
    do
        for alpha in 0.001 0.01 0.1 1.0 10.0 100.0 1000.0
        do
            alpha_part=`printf '%1.0e' $alpha`
            python train.py \
                --method sl --params beta,a=1,b=1,lam_sl=0.0001,alpha=$alpha \
                --dataset BinaryMNISTC --ds-params size=$dssize,labels=53,corruption=$CORRUPTION \
                --model LeNet \
                --max-steps $MAX_STEPS \
                --batch-size 256 \
                --mc-samples 32 \
                --outdir $OUTDIR \
                --prefix $METHOD-alpha$alpha_part
        done
    done
done
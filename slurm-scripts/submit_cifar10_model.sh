#!/bin/bash
#SBATCH --time=36:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:1
#SBATCH --exclude=dgx[1-7]
#SBATCH --array=1-5
#SBATCH --output=logs/job-%A-%a.out
#SBATCH --error=logs/job-%A-%a.err

module purge
module load miniconda
source activate bayes

export DISABLE_PBAR=1

OUTDIR="zoo/multiclass/sl"
MAX_STEPS=3000
METHOD="sl"


# for alpha in 0.01 0.1 1.0 10.0 100.0 0.05 0.5 5.0 25.0 50.0 75.0
for alpha in 0.01 0.1 1.0 10.0
do
    alpha_part=`printf '%1.0e' $alpha`

    python train.py \
        --method $METHOD --params alpha=$alpha \
        --dataset CIFAR10 --transform normalize_x_cifar \
        --model VGG11 \
        --max-steps $MAX_STEPS \
        --batch-size 256 \
        --mc-samples 32 \
        --outdir $OUTDIR \
        --prefix $METHOD-alpha$alpha_part-$SLURM_ARRAY_TASK_ID \
        --seed $SLURM_ARRAY_TASK_ID
done


# RUN LS
OUTDIR="zoo/multiclass/ls"
MAX_STEPS=3000
METHOD="ls"

python train.py \
    --method $METHOD --params smoothing=0.05 \
    --dataset CIFAR10 --transform normalize_x_cifar \
    --model VGG11 \
    --max-steps $MAX_STEPS \
    --batch-size 256 \
    --mc-samples 32 \
    --outdir $OUTDIR \
    --prefix $METHOD-$SLURM_ARRAY_TASK_ID \
    --seed $SLURM_ARRAY_TASK_ID


# RUN MFVI
OUTDIR="zoo/multiclass/mfvi"
MAX_STEPS=3000
METHOD="mfvi"

python train.py \
    --method $METHOD \
    --dataset CIFAR10 --transform normalize_x_cifar \
    --model VGG11 \
    --max-steps $MAX_STEPS \
    --batch-size 256 \
    --mc-samples 32 \
    --outdir $OUTDIR \
    --prefix $METHOD-$SLURM_ARRAY_TASK_ID \
    --seed $SLURM_ARRAY_TASK_ID

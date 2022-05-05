#!/bin/bash
#SBATCH --time=24:00:00
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

OUTDIR="zoo/abl-alpha100-unibin"
MAX_STEPS=3000
METHOD="sl"


for lam in 0.000001 0.00001 0.0001 0.001 0.01 0.1 1.0
do
    lam_part=`printf '%1.0e' $lam`
    python train.py \
        --method sl --params lam_sl=$lam,alpha=100 \
        --dataset CIFAR10 --transform normalize_x_cifar \
        --model VGG11 \
        --max-steps $MAX_STEPS \
        --batch-size 256 \
        --mc-samples 32 \
        --outdir $OUTDIR \
        --prefix $METHOD-lam$lam_part-$SLURM_ARRAY_TASK_ID \
        --seed $SLURM_ARRAY_TASK_ID
done


# RUN MFVI
OUTDIR="zoo/abl-alpha100-unibin-mfvi"
MAX_STEPS=3000
METHOD="mfvi"

python train.py \
    --method mfvi \
    --dataset CIFAR10 --transform normalize_x_cifar \
    --model VGG11 \
    --max-steps $MAX_STEPS \
    --batch-size 256 \
    --mc-samples 32 \
    --outdir $OUTDIR \
    --prefix $METHOD-$SLURM_ARRAY_TASK_ID \
    --seed $SLURM_ARRAY_TASK_ID

#!/bin/bash
#SBATCH --time=9:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=dgx[1-7]
#SBATCH --array=1-5
#SBATCH --output=logs/job-%A-%a.out
#SBATCH --error=logs/job-%A-%a.err

module purge
module load miniconda
source activate bayes

export DISABLE_PBAR=1

# general config
MAX_STEPS=5000

# RUN SL
OUTDIR="zoo/multiclass/sl"
METHOD="sl"
for alpha in 500 1000 5000 10000
do
    alpha_part=`printf '%1.0e' $alpha`

    python train.py \
        --method $METHOD --params alpha=$alpha \
        --dataset CIFAR10 --transform normalize_x_cifar_v2 \
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
METHOD="ls"
python train.py \
    --method $METHOD --params smoothing=0.01 \
    --dataset CIFAR10 --transform normalize_x_cifar_v2 \
    --model VGG11 \
    --max-steps $MAX_STEPS \
    --batch-size 256 \
    --mc-samples 32 \
    --outdir $OUTDIR \
    --prefix $METHOD-$SLURM_ARRAY_TASK_ID \
    --seed $SLURM_ARRAY_TASK_ID


# # RUN MFVI
# OUTDIR="zoo/multiclass/mfvi"
# METHOD="mfvi"
# python train.py \
#     --method $METHOD \
#     --dataset CIFAR10 --transform normalize_x_cifar_v2 \
#     --model VGG11 \
#     --max-steps $MAX_STEPS \
#     --batch-size 256 \
#     --mc-samples 32 \
#     --outdir $OUTDIR \
#     --prefix $METHOD-$SLURM_ARRAY_TASK_ID \
#     --seed $SLURM_ARRAY_TASK_ID


# # EDL
# OUTDIR="zoo/multiclass/edl"
# METHOD="edl"
# python train.py \
#     --method $METHOD --params annealing_step=1000 \
#     --dataset CIFAR10 --transform normalize_x_cifar_v2 \
#     --model VGG11EDL \
#     --max-steps $MAX_STEPS \
#     --batch-size 256 \
#     --outdir $OUTDIR \
#     --prefix $METHOD-$SLURM_ARRAY_TASK_ID \
#     --seed $SLURM_ARRAY_TASK_ID


# # EDL + computed
# OUTDIR="zoo/multiclass/edl/computed-prior"
# MAX_STEPS=5000
# METHOD="edl"
# python train.py \
#     --method $METHOD \
#     --params evidence_prior=computed,annealing_step=1000 \
#     --dataset CIFAR10 --transform normalize_x_cifar_v2 \
#     --model VGG11EDL \
#     --max-steps $MAX_STEPS \
#     --batch-size 256 \
#     --mc-samples 32 \
#     --outdir $OUTDIR \
#     --prefix $METHOD-$SLURM_ARRAY_TASK_ID


# # EDL + 0.1
# OUTDIR="zoo/multiclass/edl/skewed-prior"
# MAX_STEPS=5000
# METHOD="edl"
# python train.py \
#     --method $METHOD \
#     --params evidence_prior=0.1,annealing_step=1000 \
#     --dataset CIFAR10 --transform normalize_x_cifar_v2 \
#     --model VGG11EDL \
#     --max-steps $MAX_STEPS \
#     --batch-size 256 \
#     --mc-samples 32 \
#     --outdir $OUTDIR \
#     --prefix $METHOD-$SLURM_ARRAY_TASK_ID

#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --exclude=dgx[1-7]
#SBATCH --array=1-5
#SBATCH --output=logs/job-%A-%a.out
#SBATCH --error=logs/job-%A-%a.err

module purge
module load miniconda
source activate bdl

export DISABLE_PBAR=1

OUTDIR="zoo/"
MAX_STEPS=2000
METHOD="mfvi"

# ==========================================================================================

python train.py \
    --method $METHOD \
    --dataset BinaryMNISTC --ds-params labels=53,corruption=identity \
    --model LeNet \
    --max-steps $MAX_STEPS \
    --batch-size 256 \
    --mc-samples 32 \
    --outdir $OUTDIR \
    --prefix $METHOD-nw-$SLURM_ARRAY_TASK_ID

python train.py \
    --method $METHOD --params lam_kl=0.001 \
    --dataset BinaryMNISTC --ds-params labels=53,corruption=identity \
    --model LeNet \
    --max-steps $MAX_STEPS \
    --batch-size 256 \
    --mc-samples 32 \
    --outdir $OUTDIR \
    --prefix $METHOD-lamkl00010-nw-$SLURM_ARRAY_TASK_ID

python train.py \
    --method $METHOD \
    --dataset BinaryMNISTC --ds-params labels=53,corruption=identity \
    --model LeNet \
    --max-steps $MAX_STEPS \
    --batch-size 256 \
    --mc-samples 32 \
    --wt-loss \
    --outdir $OUTDIR \
    --prefix $METHOD-wt-$SLURM_ARRAY_TASK_ID

python train.py \
    --method $METHOD --params lam_kl=0.001 \
    --dataset BinaryMNISTC --ds-params labels=53,corruption=identity \
    --model LeNet \
    --max-steps $MAX_STEPS \
    --batch-size 256 \
    --mc-samples 32 \
    --wt-loss \
    --outdir $OUTDIR \
    --prefix $METHOD-lamkl00010-wt-$SLURM_ARRAY_TASK_ID
# ==========================================================================================

python train.py \
    --method $METHOD \
    --dataset BinaryMNISTC --ds-params labels=53,corruption=identity,imbalance=0.01 \
    --model LeNet \
    --max-steps $MAX_STEPS \
    --batch-size 256 \
    --mc-samples 32 \
    --outdir $OUTDIR \
    --prefix $METHOD-nw-$SLURM_ARRAY_TASK_ID

python train.py \
    --method $METHOD --params lam_kl=0.001 \
    --dataset BinaryMNISTC --ds-params labels=53,corruption=identity,imbalance=0.01 \
    --model LeNet \
    --max-steps $MAX_STEPS \
    --batch-size 256 \
    --mc-samples 32 \
    --outdir $OUTDIR \
    --prefix $METHOD-lamkl00010-nw-$SLURM_ARRAY_TASK_ID

python train.py \
    --method $METHOD \
    --dataset BinaryMNISTC --ds-params labels=53,corruption=identity,imbalance=0.01 \
    --model LeNet \
    --max-steps $MAX_STEPS \
    --batch-size 256 \
    --mc-samples 32 \
    --wt-loss \
    --outdir $OUTDIR \
    --prefix $METHOD-wt-$SLURM_ARRAY_TASK_ID

python train.py \
    --method $METHOD --params lam_kl=0.001 \
    --dataset BinaryMNISTC --ds-params labels=53,corruption=identity,imbalance=0.01 \
    --model LeNet \
    --max-steps $MAX_STEPS \
    --batch-size 256 \
    --mc-samples 32 \
    --wt-loss \
    --outdir $OUTDIR \
    --prefix $METHOD-lamkl00010-wt-$SLURM_ARRAY_TASK_ID
# ==========================================================================================

python train.py \
    --method $METHOD \
    --dataset BinaryMNISTC --ds-params labels=53,corruption=identity,imbalance=0.005 \
    --model LeNet \
    --max-steps $MAX_STEPS \
    --batch-size 256 \
    --mc-samples 32 \
    --outdir $OUTDIR \
    --prefix $METHOD-nw-$SLURM_ARRAY_TASK_ID

python train.py \
    --method $METHOD --params lam_kl=0.001 \
    --dataset BinaryMNISTC --ds-params labels=53,corruption=identity,imbalance=0.005 \
    --model LeNet \
    --max-steps $MAX_STEPS \
    --batch-size 256 \
    --mc-samples 32 \
    --outdir $OUTDIR \
    --prefix $METHOD-lamkl00010-nw-$SLURM_ARRAY_TASK_ID

python train.py \
    --method $METHOD \
    --dataset BinaryMNISTC --ds-params labels=53,corruption=identity,imbalance=0.005 \
    --model LeNet \
    --max-steps $MAX_STEPS \
    --batch-size 256 \
    --mc-samples 32 \
    --wt-loss \
    --outdir $OUTDIR \
    --prefix $METHOD-wt-$SLURM_ARRAY_TASK_ID

python train.py \
    --method $METHOD --params lam_kl=0.001 \
    --dataset BinaryMNISTC --ds-params labels=53,corruption=identity,imbalance=0.005 \
    --model LeNet \
    --max-steps $MAX_STEPS \
    --batch-size 256 \
    --mc-samples 32 \
    --wt-loss \
    --outdir $OUTDIR \
    --prefix $METHOD-lamkl00010-wt-$SLURM_ARRAY_TASK_ID
# ==========================================================================================

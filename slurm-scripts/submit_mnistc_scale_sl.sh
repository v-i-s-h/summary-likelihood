#!/bin/bash
#SBATCH --time=3:00:00
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

OUTDIR="zoo/sl-scaling"
MAX_STEPS=2000
METHOD="sl"

python train.py \
    --method $METHOD --params auto,ea=0.98,lam_sl=0.00001 \
    --dataset BinaryMNISTC --ds-params labels=53,corruption=identity \
    --model LeNet \
    --max-steps $MAX_STEPS \
    --batch-size 256 \
    --mc-samples 32 \
    --outdir $OUTDIR \
    --prefix $METHOD-lam1e-5-nw-$SLURM_ARRAY_TASK_ID

python train.py \
    --method $METHOD --params auto,ea=0.98,lam_sl=0.0001 \
    --dataset BinaryMNISTC --ds-params labels=53,corruption=identity \
    --model LeNet \
    --max-steps $MAX_STEPS \
    --batch-size 256 \
    --mc-samples 32 \
    --outdir $OUTDIR \
    --prefix $METHOD-lam1e-4-nw-$SLURM_ARRAY_TASK_ID

python train.py \
    --method $METHOD --params auto,ea=0.98,lam_sl=0.001 \
    --dataset BinaryMNISTC --ds-params labels=53,corruption=identity \
    --model LeNet \
    --max-steps $MAX_STEPS \
    --batch-size 256 \
    --mc-samples 32 \
    --outdir $OUTDIR \
    --prefix $METHOD-lam1e-3-nw-$SLURM_ARRAY_TASK_ID

python train.py \
    --method $METHOD --params auto,ea=0.98,lam_sl=0.01 \
    --dataset BinaryMNISTC --ds-params labels=53,corruption=identity \
    --model LeNet \
    --max-steps $MAX_STEPS \
    --batch-size 256 \
    --mc-samples 32 \
    --outdir $OUTDIR \
    --prefix $METHOD-lam1e-2-nw-$SLURM_ARRAY_TASK_ID

python train.py \
    --method $METHOD --params auto,ea=0.98,lam_sl=0.1 \
    --dataset BinaryMNISTC --ds-params labels=53,corruption=identity \
    --model LeNet \
    --max-steps $MAX_STEPS \
    --batch-size 256 \
    --mc-samples 32 \
    --outdir $OUTDIR \
    --prefix $METHOD-lam1e-1-nw-$SLURM_ARRAY_TASK_ID

python train.py \
    --method $METHOD --params auto,ea=0.98,lam_sl=1.0 \
    --dataset BinaryMNISTC --ds-params labels=53,corruption=identity \
    --model LeNet \
    --max-steps $MAX_STEPS \
    --batch-size 256 \
    --mc-samples 32 \
    --outdir $OUTDIR \
    --prefix $METHOD-lam1e+0-nw-$SLURM_ARRAY_TASK_ID

python train.py \
    --method $METHOD --params auto,ea=0.98,lam_sl=10.0 \
    --dataset BinaryMNISTC --ds-params labels=53,corruption=identity \
    --model LeNet \
    --max-steps $MAX_STEPS \
    --batch-size 256 \
    --mc-samples 32 \
    --outdir $OUTDIR \
    --prefix $METHOD-lam1e+1-nw-$SLURM_ARRAY_TASK_ID

python train.py \
    --method $METHOD --params auto,ea=0.98,lam_sl=100.0 \
    --dataset BinaryMNISTC --ds-params labels=53,corruption=identity \
    --model LeNet \
    --max-steps $MAX_STEPS \
    --batch-size 256 \
    --mc-samples 32 \
    --outdir $OUTDIR \
    --prefix $METHOD-lam1e+2-nw-$SLURM_ARRAY_TASK_ID

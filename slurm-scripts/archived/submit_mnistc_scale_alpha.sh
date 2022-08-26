#!/bin/bash
#SBATCH --time=2:00:00
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

OUTDIR="zoo/alpha-scaling-lam1.0/"
MAX_STEPS=2000
METHOD="sl"

python train.py \
    --method $METHOD --params auto,ea=0.98,alpha=0.001,lam_sl=1.0 \
    --dataset BinaryMNISTC --ds-params labels=53,corruption=identity \
    --model LeNet \
    --max-steps $MAX_STEPS \
    --batch-size 256 \
    --mc-samples 32 \
    --outdir $OUTDIR \
    --prefix $METHOD-1e-3-nw-$SLURM_ARRAY_TASK_ID

python train.py \
    --method $METHOD --params auto,ea=0.98,alpha=0.005,lam_sl=1.0 \
    --dataset BinaryMNISTC --ds-params labels=53,corruption=identity \
    --model LeNet \
    --max-steps $MAX_STEPS \
    --batch-size 256 \
    --mc-samples 32 \
    --outdir $OUTDIR \
    --prefix $METHOD-5e-3-nw-$SLURM_ARRAY_TASK_ID

python train.py \
    --method $METHOD --params auto,ea=0.98,alpha=0.01,lam_sl=1.0 \
    --dataset BinaryMNISTC --ds-params labels=53,corruption=identity \
    --model LeNet \
    --max-steps $MAX_STEPS \
    --batch-size 256 \
    --mc-samples 32 \
    --outdir $OUTDIR \
    --prefix $METHOD-1e-2-nw-$SLURM_ARRAY_TASK_ID

python train.py \
    --method $METHOD --params auto,ea=0.98,alpha=0.05,lam_sl=1.0 \
    --dataset BinaryMNISTC --ds-params labels=53,corruption=identity \
    --model LeNet \
    --max-steps $MAX_STEPS \
    --batch-size 256 \
    --mc-samples 32 \
    --outdir $OUTDIR \
    --prefix $METHOD-5e-2-nw-$SLURM_ARRAY_TASK_ID

python train.py \
    --method $METHOD --params auto,ea=0.98,alpha=0.1,lam_sl=1.0 \
    --dataset BinaryMNISTC --ds-params labels=53,corruption=identity \
    --model LeNet \
    --max-steps $MAX_STEPS \
    --batch-size 256 \
    --mc-samples 32 \
    --outdir $OUTDIR \
    --prefix $METHOD-1e-1-nw-$SLURM_ARRAY_TASK_ID

python train.py \
    --method $METHOD --params auto,ea=0.98,alpha=0.5,lam_sl=1.0 \
    --dataset BinaryMNISTC --ds-params labels=53,corruption=identity \
    --model LeNet \
    --max-steps $MAX_STEPS \
    --batch-size 256 \
    --mc-samples 32 \
    --outdir $OUTDIR \
    --prefix $METHOD-5e-1-nw-$SLURM_ARRAY_TASK_ID

python train.py \
    --method $METHOD --params auto,ea=0.98,alpha=1.0,lam_sl=1.0 \
    --dataset BinaryMNISTC --ds-params labels=53,corruption=identity \
    --model LeNet \
    --max-steps $MAX_STEPS \
    --batch-size 256 \
    --mc-samples 32 \
    --outdir $OUTDIR \
    --prefix $METHOD-1e+0-nw-$SLURM_ARRAY_TASK_ID

python train.py \
    --method $METHOD --params auto,ea=0.98,alpha=5.0,lam_sl=1.0 \
    --dataset BinaryMNISTC --ds-params labels=53,corruption=identity \
    --model LeNet \
    --max-steps $MAX_STEPS \
    --batch-size 256 \
    --mc-samples 32 \
    --outdir $OUTDIR \
    --prefix $METHOD-5e+0-nw-$SLURM_ARRAY_TASK_ID

python train.py \
    --method $METHOD --params auto,ea=0.98,alpha=10.0,lam_sl=1.0 \
    --dataset BinaryMNISTC --ds-params labels=53,corruption=identity \
    --model LeNet \
    --max-steps $MAX_STEPS \
    --batch-size 256 \
    --mc-samples 32 \
    --outdir $OUTDIR \
    --prefix $METHOD-1e+1-nw-$SLURM_ARRAY_TASK_ID

python train.py \
    --method $METHOD --params auto,ea=0.98,alpha=50.0,lam_sl=1.0 \
    --dataset BinaryMNISTC --ds-params labels=53,corruption=identity \
    --model LeNet \
    --max-steps $MAX_STEPS \
    --batch-size 256 \
    --mc-samples 32 \
    --outdir $OUTDIR \
    --prefix $METHOD-5e+1-nw-$SLURM_ARRAY_TASK_ID


python train.py \
    --method $METHOD --params auto,ea=0.98,alpha=100.0,lam_sl=1.0 \
    --dataset BinaryMNISTC --ds-params labels=53,corruption=identity \
    --model LeNet \
    --max-steps $MAX_STEPS \
    --batch-size 256 \
    --mc-samples 32 \
    --outdir $OUTDIR \
    --prefix $METHOD-1e+2-nw-$SLURM_ARRAY_TASK_ID

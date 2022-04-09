#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --exclude=dgx[1-7]
#SBATCH --array=1-5
#SBATCH --output=logs/job-%A-%a.out
#SBATCH --error=logs/job-%A-%a.err

module purge
module load miniconda
source activate bayes

export DISABLE_PBAR=1

OUTDIR="zoo/MedMNIST/"
MAX_STEPS=30000

# ==========================================================================================
METHOD="mfvi"
python train.py \
    --method $METHOD \
    --dataset ChestMNIST --ds-params label=infiltration \
    --model ConvNet \
    --max-steps $MAX_STEPS \
    --batch-size 256 \
    --mc-samples 32 \
    --wt-loss \
    --outdir $OUTDIR \
    --prefix $METHOD-wt-$SLURM_ARRAY_TASK_ID

#
METHOD="sl"
python train.py \
    --method $METHOD --params auto,ea=0.75 \
    --dataset ChestMNIST --ds-params label=infiltration \
    --model ConvNet \
    --max-steps $MAX_STEPS \
    --batch-size 256 \
    --mc-samples 32 \
    --wt-loss \
    --outdir $OUTDIR \
    --prefix $METHOD-auto-wt-$SLURM_ARRAY_TASK_ID

python train.py \
    --method $METHOD --params auto,ea=0.75,lam_kl=0.001 \
    --dataset ChestMNIST --ds-params label=infiltration \
    --model ConvNet \
    --max-steps $MAX_STEPS \
    --batch-size 256 \
    --mc-samples 32 \
    --wt-loss \
    --outdir $OUTDIR \
    --prefix $METHOD-auto-lamkl00010-wt-$SLURM_ARRAY_TASK_ID
# ==========================================================================================

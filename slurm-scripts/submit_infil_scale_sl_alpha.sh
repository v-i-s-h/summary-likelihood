#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH --exclude=dgx[1-7]
#SBATCH --array=1-5
#SBATCH --output=logs/job-%A-%a.out
#SBATCH --error=logs/job-%A-%a.err

module purge
module load miniconda
source activate bdl

export DISABLE_PBAR=1

OUTDIR="zoo/alpha-sl-scaling"
MAX_STEPS=30000
METHOD="sl"

# ------------------------------------------------------------------------------

for alpha in 1.0 5.0 10.0
do
    for lam in 0.00001 0.0001 0.001 0.01 0.1
    do
        lam_part=`printf '%1.0e' $lam`
        alpha_part=`printf '%1.0e' $alpha`
        
        python train.py \
            --method $METHOD --params auto,ea=0.75,lam_sl=$lam,alpha=$alpha \
            --dataset ChestMNIST --ds-params label=infiltration \
            --model ConvNet \
            --max-steps $MAX_STEPS \
            --batch-size 256 \
            --mc-samples 32 \
            --wt-loss \
            --outdir $OUTDIR \
            --prefix $METHOD-wt-lam$lam_part-alpha$alpha_part-auto-ea075-$SLURM_ARRAY_TASK_ID

        python train.py \
            --method $METHOD --params beta,a=2,b=4,lam_sl=$lam,alpha=$alpha \
            --dataset ChestMNIST --ds-params label=infiltration \
            --model ConvNet \
            --max-steps $MAX_STEPS \
            --batch-size 256 \
            --mc-samples 32 \
            --wt-loss \
            --outdir $OUTDIR \
            --prefix $METHOD-wt-lam$lam_part-alpha$alpha_part-beta-2-4-$SLURM_ARRAY_TASK_ID
    done
done
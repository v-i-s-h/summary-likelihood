#!/bin/bash
#SBATCH --time=16:00:00
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

OUTDIR="zoo/alpha-sl-scaling"
MAX_STEPS=2000
METHOD="sl"

# ------------------------------------------------------------------------------

for alpha in 0.001 0.005 0.01 0.05 0.1 0.5 1.0 5.0 10.0 50.0 100.0
do
    for lam in 0.00001 0.0001 0.001 0.01 0.1 1.0 10.0 100.0
    do
        lam_part=`printf '%1.0e' $lam`
        alpha_part=`printf '%1.0e' $alpha`
        python train.py \
            --method $METHOD --params auto,ea=0.98,lam_sl=$lam,alpha=$alpha \
            --dataset BinaryMNISTC --ds-params labels=53,corruption=identity \
            --model LeNet \
            --max-steps $MAX_STEPS \
            --batch-size 256 \
            --mc-samples 32 \
            --outdir $OUTDIR \
            --prefix $METHOD-lam$lam_part-alpha$alpha_part-nw-$SLURM_ARRAY_TASK_ID
    done
done
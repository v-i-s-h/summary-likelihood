#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=dgx[1-7]
#SBATCH --array=1-5
#SBATCH --output=logs/job-%A-%a.out
#SBATCH --error=logs/job-%A-%a.err

module purge
module load miniconda
source activate bayes

export DISABLE_PBAR=1

OUTDIR="zoo/abl-a100-mnist-a050b050"
MAX_STEPS=3000
METHOD="sl"

for dssize in 10000 1000
do
    for lam in 0.000001 0.00001 0.0001 0.001 0.01 0.1 1.0
    do
        lam_part=`printf '%1.0e' $lam`
        python train.py \
            --method $METHOD --params beta,a=0.5,b=0.5,lam_sl=$lam,alpha=100 \
            --dataset BinaryMNISTC --ds-params labels=53,corruption=identity \
            --model LeNet \
            --max-steps $MAX_STEPS \
            --batch-size 256 \
            --mc-samples 32 \
            --outdir $OUTDIR \
            --prefix $METHOD-lam$lam_part-sz$dssize-$SLURM_ARRAY_TASK_ID
    done
done


for dssize in 10000 1000
do
    for lam in 0.000001 0.00001 0.0001 0.001 0.01 0.1 1.0
    do
        lam_part=`printf '%1.0e' $lam`
        python train.py \
            --method $METHOD --params beta,a=0.5,b=0.5,lam_sl=$lam,alpha=100 \
            --dataset BinaryMNISTC --ds-params labels=53,corruption=identity \
            --model ConvNet \
            --max-steps $MAX_STEPS \
            --batch-size 256 \
            --mc-samples 32 \
            --outdir $OUTDIR \
            --prefix $METHOD-lam$lam_part-sz$dssize-$SLURM_ARRAY_TASK_ID
    done
done
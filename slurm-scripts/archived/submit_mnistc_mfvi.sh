#!/bin/bash
#SBATCH --time=0:45:00
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

OUTDIR="zoo/bmnist53-lenet"
MAX_STEPS=3000
METHOD="mfvi"

for dssize in 10000 1000
do
    python train.py \
        --method $METHOD \
        --dataset BinaryMNISTC --ds-params size=$dssize,labels=53,corruption=identity \
        --model LeNet \
        --max-steps $MAX_STEPS \
        --batch-size 256 \
        --mc-samples 32 \
        --outdir $OUTDIR \
        --prefix $METHOD-sz$dssize-$SLURM_ARRAY_TASK_ID
done



# OUTDIR="zoo/bmnist53-convnet"
# MAX_STEPS=3000
# METHOD="mfvi"

# for dssize in 10000 1000
# do
#     python train.py \
#         --method $METHOD \
#         --dataset BinaryMNISTC --ds-params size=$dssize,labels=53,corruption=identity \
#         --model ConvNet \
#         --max-steps $MAX_STEPS \
#         --batch-size 256 \
#         --mc-samples 32 \
#         --outdir $OUTDIR \
#         --prefix $METHOD-sz$dssize-$SLURM_ARRAY_TASK_ID
# done

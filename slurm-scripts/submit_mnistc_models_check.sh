#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --exclude=dgx[1-7]
#SBATCH --array=1-3
#SBATCH --output=logs/job-%A-%a.out
#SBATCH --error=logs/job-%A-%a.err

module purge
module load miniconda
source activate bayes

export DISABLE_PBAR=1


# ------------------------------ SL (uniform prior) ----------------------------
OUTDIR="zoo/sl_check/uniform-prior-alphavar"
MAX_STEPS=3000
METHOD="sl"

for alpha in 200.0 250.0 500.0 750.0
do
    for dssize in 8000
    do
        alpha_part=`printf '%1.0e' $alpha`

        # LeNet
        python train.py \
            --method $METHOD --params beta,a=1,b=1,alpha=$alpha \
            --dataset BinaryMNISTC \
            --ds-params size=$dssize,labels=53,corruption=identity \
            --model LeNet \
            --max-steps $MAX_STEPS \
            --batch-size 256 \
            --mc-samples 32 \
            --outdir $OUTDIR \
            --prefix $METHOD-alpha$alpha_part-sz$dssize-$SLURM_ARRAY_TASK_ID

        # # ConvNet
        # python train.py \
        #     --method $METHOD --params beta,a=1,b=1,alpha=$alpha \
        #     --dataset BinaryMNISTC \
        #     --ds-params size=$dssize,labels=53,corruption=identity \
        #     --model ConvNet \
        #     --max-steps $MAX_STEPS \
        #     --batch-size 256 \
        #     --mc-samples 32 \
        #     --outdir $OUTDIR \
        #     --prefix $METHOD-alpha$alpha_part-sz$dssize-$SLURM_ARRAY_TASK_ID
    done
done

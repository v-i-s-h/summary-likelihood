#!/bin/bash
#SBATCH --time=0:20:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --exclude=dgx[1-7]
#SBATCH --array=1-3
#SBATCH --output=logs/fig1-job-%A-%a.out
#SBATCH --error=logs/fig1-job-%A-%a.err

module purge
module load miniconda
source activate bayes

export DISABLE_PBAR=1


OUTDIR="zoo/sl-effect/"
MAX_STEPS=2000
METHOD="sl"
alpha=10000
alpha_part=`printf '%1.0e' $alpha`


case $SLURM_ARRAY_TASK_ID in
    1)  
        a=0.1
        b=0.1
    ;;
    2)  
        a=1.0
        b=1.0
    ;;
    3)  
        a=5.0
        b=5.0
    ;;
    *) 
        a=1.0
        b=1.0
    ;;
esac

signature=`printf 'beta-%1.0e-%1.0e' $a $b`

python train.py \
    --method sl --params beta,a=$a,b=$b,alpha=$alpha \
    --dataset BinaryMNISTC \
    --ds-params size=1000,labels=53,corruption=identity \
    --model LeNet \
    --max-steps $MAX_STEPS \
    --batch-size 256 \
    --mc-samples 32 \
    --outdir $OUTDIR \
    --prefix $METHOD-alpha$alpha_part-$signature \
    --seed 42


#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:1
#SBATCH --exclude=dgx[1-7]
#SBATCH --output=logs/job-%A.out
#SBATCH --error=logs/job-%A.err

module purge
module load miniconda
source activate bayes

export DISABLE_PBAR=1


echo "========================= MFVI: CIFAR10 + VGG11 =========================="
OUTDIR="zoo/multiclass/mfvi/CIFAR10/VGG11"
python eval_calib.py \
    --models  ${OUTDIR}/mfvi-*
echo "=========================================================================="


echo "======================== LS: CIFAR10 + VGG11 ========================="
OUTDIR="zoo/multiclass/ls/CIFAR10/VGG11"
python eval_calib.py \
    --models  ${OUTDIR}/ls-*
echo "=========================================================================="


echo "======================== SL: CIFAR10 + VGG11 ========================="
OUTDIR="zoo/multiclass/sl/CIFAR10/VGG11"
for alpha in 0.01 0.1 1.0 10.0
do
    alpha_part=`printf '%1.0e' $alpha`

    python eval_calib.py \
        --models  ${basedir}/sl-alpha$alpha_part-*
    echo "-----------------------------------------------------------------"
done
echo "=========================================================================="
#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --mem-per-cpu=6G
#SBATCH --gres=gpu:1
#SBATCH --exclude=dgx[1-7]
#SBATCH --output=logs/job-%A.out
#SBATCH --error=logs/job-%A.err

module purge
module load miniconda
source activate bayes

export DISABLE_PBAR=1

OOD_DATASET="SVHN"


echo "========================= SL: CIFAR10 + VGG11 =========================="
OUTDIR="zoo/multiclass/slim/CIFAR10Im/VGG11"
for alpha in 500 1000 5000 10000
do
    alpha_part=`printf '%1.0e' $alpha`

    echo "--------------------- SL: alpha = ${alpha_part}------------------"
    python eval_ood.py \
        --ood $OOD_DATASET \
        --models $OUTDIR/slim-alpha$alpha_part-*
done
echo "=========================================================================="


echo "========================= MFVI: CIFAR10 + VGG11 =========================="
OUTDIR="zoo/multiclass/mfvi/CIFAR10Im/VGG11"
python eval_ood.py \
    --ood $OOD_DATASET \
    --models $OUTDIR/mfvi-*
echo "=========================================================================="


echo "======================== LS: CIFAR10 + VGG11 ========================="
OUTDIR="zoo/multiclass/ls/CIFAR10Im/VGG11"
python eval_ood.py \
    --ood $OOD_DATASET \
    --models $OUTDIR/ls-*
echo "========================================================================="


echo "================================= EDL ==================================="
OUTDIR="zoo/multiclass/edl/uniform-prior/CIFAR10Im/VGG11EDL"
python eval_ood.py \
    --ood $OOD_DATASET \
    --models $OUTDIR/edl-*
# echo "========================================================================="

echo "================================= EDL ==================================="
OUTDIR="zoo/multiclass/edl/computed-prior/CIFAR10Im/VGG11EDL"
python eval_ood.py \
    --ood $OOD_DATASET \
    --models $OUTDIR/edl-*
echo "========================================================================="

#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --mem-per-cpu=6G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/job-%A.out
#SBATCH --error=logs/job-%A.err

module purge
module load miniconda
source activate bayes

export DISABLE_PBAR=1

OOD_DATASET="SVHN"


# echo "========================= SL: CIFAR10 + VGG11 =========================="
# OUTDIR="zoo/multiclass/sl/CIFAR10/VGG11"
# for alpha in 500 1000 5000 10000
# do
#     alpha_part=`printf '%1.0e' $alpha`

#     echo "--------------------- SL: alpha = ${alpha_part}------------------"
#     python eval_ood.py \
#         --ood $OOD_DATASET \
#         --models $OUTDIR/sl-alpha$alpha_part-*
# done
# echo "=========================================================================="


# echo "========================= MFVI: CIFAR10 + VGG11 =========================="
# OUTDIR="zoo/multiclass/mfvi/CIFAR10/VGG11"
# python eval_ood.py \
#     --ood $OOD_DATASET \
#     --models $OUTDIR/mfvi-*
# echo "=========================================================================="


# echo "======================== LS: CIFAR10 + VGG11 ========================="
# OUTDIR="zoo/multiclass/ls/CIFAR10/VGG11"
# python eval_ood.py \
#     --ood $OOD_DATASET \
#     --models $OUTDIR/ls-*
# echo "========================================================================="


# echo "================================= EDL ==================================="
# OUTDIR="zoo/multiclass/edl/CIFAR10/VGG11EDL"
# python eval_ood.py \
#     --ood $OOD_DATASET \
#     --models $OUTDIR/edl-*
# echo "========================================================================="

# # echo "================================= EDL ==================================="
# # OUTDIR="zoo/multiclass/edl/computed-prior/CIFAR10/VGG11EDL"
# # python eval_ood.py \
# #     --ood $OOD_DATASET \
# #     --models $OUTDIR/edl-*
# # echo "========================================================================="

# # echo "================================= EDL ==================================="
# # OUTDIR="zoo/multiclass/edl/skewed-prior/CIFAR10/VGG11EDL"
# # python eval_ood.py \
# #     --ood $OOD_DATASET \
# #     --models $OUTDIR/edl-*
# # echo "========================================================================="


echo "======================== SGD-X CIFAR10 + VGG11 =========================="
OUTDIR="zoo/sgd-rebuttal"

python eval_ood.py \
    --ood $OOD_DATASET \
    --models  ${OUTDIR}/5k/noaug/CIFAR10/VGG11Deterministic/sgd-noaug-*

python eval_ood.py \
    --ood $OOD_DATASET \
    --models  ${OUTDIR}/30k/noaug/CIFAR10/VGG11Deterministic/sgd-noaug-*

python eval_ood.py \
    --ood $OOD_DATASET \
    --models  ${OUTDIR}/5k/aug/CIFAR10/VGG11Deterministic/sgd-da-*

python eval_ood.py \
    --ood $OOD_DATASET \
    --models  ${OUTDIR}/30k/aug/CIFAR10/VGG11Deterministic/sgd-da-*

for alpha in 1000
do
    alpha_part=`printf '%1.0e' $alpha`
    python eval_ood.py \
        --ood $OOD_DATASET \
        --models  ${OUTDIR}/5k/noaug/CIFAR10/VGG11Deterministic/sgdsl-noaug-alpha$alpha_part-*

    python eval_ood.py \
        --ood $OOD_DATASET \
        --models  ${OUTDIR}/30k/noaug/CIFAR10/VGG11Deterministic/sgdsl-noaug-alpha$alpha_part-*

    python eval_ood.py \
        --ood $OOD_DATASET \
        --models  ${OUTDIR}/5k/aug/CIFAR10/VGG11Deterministic/sgdsl-da-alpha$alpha_part-*

    python eval_ood.py \
        --ood $OOD_DATASET \
        --models  ${OUTDIR}/30k/aug/CIFAR10/VGG11Deterministic/sgdsl-da-alpha$alpha_part-*
done
echo "========================================================================="
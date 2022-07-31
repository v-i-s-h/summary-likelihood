#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=dgx[1-7]
#SBATCH --output=logs/job-%A.out
#SBATCH --error=logs/job-%A.err

module purge
module load miniconda
source activate bayes

export DISABLE_PBAR=1

# echo "======================== MFVI: CIFAR10 + VGG11 ========================="
# basedir="zoo/abl-alpha100-unibin-mfvi-cifar10/CIFAR10/VGG11/"
# python eval_calib.py \
#     --models  ${basedir}/mfvi-*
# echo "=========================================================================="

echo "======================== LS: CIFAR10 + VGG11 ========================="
basedir="zoo/abl-alpha100-cifar10-ls/CIFAR10/VGG11/"
python eval_calib.py \
    --models  ${basedir}/ls-*
echo "=========================================================================="

# echo "======================== SL: CIFAR10 + VGG11 ========================="
# basedir="zoo/abl-alpha100-unibin/CIFAR10/VGG11/"
# for lam in 0.000001 0.00001 0.0001 0.001 0.01 0.1 1.0
# do
#     lam_part=`printf '%1.0e' $lam`
#     python eval_calib.py \
#         --models  ${basedir}/sl-lam$lam_part-*
#     echo "-----------------------------------------------------------------"
# done
# echo "=========================================================================="

# echo "======================== SLIM: CIFAR10 + VGG11 ========================="
# basedir="zoo/abl-alpha100-slim-CIFAR10/CIFAR10/VGG11/"
# # for lam in 0.000001 0.00001 0.0001 0.001 0.01 0.1 1.0
# for lam in 1.0
# do
#     lam_part=`printf '%1.0e' $lam`
#     python eval_calib.py \
#         --models  ${basedir}/slim-lam$lam_part-*
#     echo "-----------------------------------------------------------------"
# done
# echo "=========================================================================="

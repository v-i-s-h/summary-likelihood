#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --mem-per-cpu=6G
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=dgx[1-7]
#SBATCH --array=1-20
#SBATCH --output=logs/job-%A-%a.out
#SBATCH --error=logs/job-%A-%a.err

module purge
module load miniconda
source activate bayes

export DISABLE_PBAR=1


case $SLURM_ARRAY_TASK_ID in
    1) CORRUPTION=brightness ;;
    2) CORRUPTION=contrast ;;
    3) CORRUPTION=defocus_blur ;;
    4) CORRUPTION=elastic_transform ;;
    5) CORRUPTION=fog ;;
    6) CORRUPTION=frost ;;
    7) CORRUPTION=gaussian_blur ;;
    8) CORRUPTION=gaussian_noise ;;
    9) CORRUPTION=glass_blur ;;
    10) CORRUPTION=identity ;;
    11) CORRUPTION=impulse_noise ;;
    12) CORRUPTION=jpeg_compression ;;
    13) CORRUPTION=motion_blur ;;
    14) CORRUPTION=pixelate ;;
    15) CORRUPTION=saturate ;;
    16) CORRUPTION=shot_noise ;;
    17) CORRUPTION=snow ;;
    18) CORRUPTION=spatter ;;
    19) CORRUPTION=speckle_noise ;;
    20) CORRUPTION=zoom_blur ;;
    *) CORRUPTION=identity ;;
esac


# echo "======================== SL: CIFAR10 + VGG11 ========================="
# OUTDIR="zoo/multiclass/sl/CIFAR10/VGG11"
# for alpha in 500 1000 5000 10000
# do
#     alpha_part=`printf '%1.0e' $alpha`

#     echo "--------------------- SL: alpha = ${alpha_part}------------------"
#     for level in {1..5}
#     do
#         python eval_calib.py \
#             --corruption $CORRUPTION-$level \
#             --models  ${OUTDIR}/sl-alpha$alpha_part-*
#     done
# done
# echo "=========================================================================="


# echo "========================= MFVI: CIFAR10 + VGG11 =========================="
# OUTDIR="zoo/multiclass/mfvi/CIFAR10/VGG11"
# for level in {1..5}
# do
#     python eval_calib.py \
#         --corruption $CORRUPTION-$level \
#         --models  ${OUTDIR}/mfvi-*
# done
# echo "=========================================================================="


# echo "======================== LS: CIFAR10 + VGG11 ========================="
# OUTDIR="zoo/multiclass/ls/CIFAR10/VGG11"
# for level in {1..5}
# do
#     python eval_calib.py \
#     --corruption $CORRUPTION-$level \
#     --models  ${OUTDIR}/ls-*
# done
# echo "========================================================================="


# echo "================================= EDL ==================================="
# OUTDIR="zoo/multiclass/edl/CIFAR10/VGG11EDL"
# for level in {1..5}
# do
#     python eval_calib.py \
#     --corruption $CORRUPTION-$level \
#     --models  ${OUTDIR}/edl-*
# done
# echo "========================================================================="


# echo "================================= EDL ==================================="
# OUTDIR="zoo/multiclass/edl/computed-prior/CIFAR10/VGG11EDL"
# python eval_calib.py \
#     --corruption $CORRUPTION \
#     --models  ${OUTDIR}/edl-*
# echo "========================================================================="

# echo "================================= EDL ==================================="
# OUTDIR="zoo/multiclass/edl/skewed-prior/CIFAR10/VGG11EDL"
# python eval_calib.py \
#     --corruption $CORRUPTION \
#     --models  ${OUTDIR}/edl-*
# echo "========================================================================="



# ------------------------------------------------------------------------------
# echo "================================= EDL ==================================="
# OUTDIR="zoo/multiclass-v2/edl/CIFAR10/VGG11EDL"

# for i in 1 2 3 4 5
# do
#     python eval_calib.py \
#         --corruption $CORRUPTION-$i \
#         --models  ${OUTDIR}/edl-*
# done
# echo "========================================================================="


# echo "========================= MFVI: CIFAR10 + VGG11 =========================="
# OUTDIR="zoo/multiclass-v2/mfvi/CIFAR10/VGG11"
# for i in 1 2 3 4 5
# do
#     python eval_calib.py \
#         --corruption $CORRUPTION-$i \
#         --models  ${OUTDIR}/mfvi-*
# done
# echo "=========================================================================="


# echo "======================== LS: CIFAR10 + VGG11 ========================="
# OUTDIR="zoo/multiclass-v2/ls/CIFAR10/VGG11"

# for i in 1 2 3 4 5
# do
#     python eval_calib.py \
#         --corruption $CORRUPTION-$i \
#         --models  ${OUTDIR}/ls-*
# done
# echo "========================================================================="


# echo "======================== SL: CIFAR10 + VGG11 ========================="
# OUTDIR="zoo/multiclass-v2/sl/CIFAR10/VGG11"
# for alpha in 100.0 500.0 1000.0 5000.0 10000.0 50000.0
# do
#     alpha_part=`printf '%1.0e' $alpha`

#     echo "--------------------- SL: alpha = ${alpha_part}------------------"
#     for i in 1 2 3 4 5
#     do
#         python eval_calib.py \
#             --corruption $CORRUPTION-$i \
#             --models  ${OUTDIR}/sl-alpha$alpha_part-*
#     done
# done
# echo "=========================================================================="


# ------------------------------------------------------------------------------

echo "======================== SGD-X CIFAR10 + VGG11 =========================="
OUTDIR="zoo/multiclass/sgd-rebuttal/CIFAR10/VGG11Deterministic"

for level in {1..5}
do
    python eval_calib.py \
        --corruption $CORRUPTION-$level \
        --models  ${OUTDIR}/5k/noaug/sgd-noaug-*

    python eval_calib.py \
        --corruption $CORRUPTION-$level \
        --models  ${OUTDIR}/30k/noaug/sgd-noaug-*

    python eval_calib.py \
        --corruption $CORRUPTION-$level \
        --models  ${OUTDIR}/5k/aug/sgd-da-*

    python eval_calib.py \
        --corruption $CORRUPTION-$level \
        --models  ${OUTDIR}/30k/aug/sgd-da-*

    for alpha in 1000
    do
        alpha_part=`printf '%1.0e' $alpha`
    
        python eval_calib.py \
            --corruption $CORRUPTION-$level \
            --models  ${OUTDIR}/5k/noaug/sgdsl-noaug-alpha$alpha_part-*

        python eval_calib.py \
            --corruption $CORRUPTION-$level \
            --models  ${OUTDIR}/30k/noaug/sgdsl-noaug-alpha$alpha_part-*

        python eval_calib.py \
            --corruption $CORRUPTION-$level \
            --models  ${OUTDIR}/5k/aug/sgdsl-da-alpha$alpha_part-*

        python eval_calib.py \
            --corruption $CORRUPTION-$level \
            --models  ${OUTDIR}/30k/aug/sgdsl-da-alpha$alpha_part-*
    done
done
echo "========================================================================="
#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --exclude=dgx[1-7]
#SBATCH --array=1-16
#SBATCH --output=logs/job-%A-%a.out
#SBATCH --error=logs/job-%A-%a.err

module purge
module load miniconda
source activate bayes

export DISABLE_PBAR=1


case $SLURM_ARRAY_TASK_ID in
    1) CORRUPTION=brightness ;;
    2) CORRUPTION=canny_edges ;;
    3) CORRUPTION=dotted_line ;;
    4) CORRUPTION=fog ;;
    5) CORRUPTION=glass_blur ;;
    6) CORRUPTION=identity ;;
    7) CORRUPTION=impulse_noise ;;
    8) CORRUPTION=motion_blur ;;
    9) CORRUPTION=rotate ;;
    10) CORRUPTION=scale ;;
    11) CORRUPTION=shear;;
    12) CORRUPTION=shot_noise ;;
    13) CORRUPTION=spatter ;;
    14) CORRUPTION=stripe ;;
    15) CORRUPTION=translate ;;
    16) CORRUPTION=zigzag ;;
    *) CORRUPTION=identity ;;
esac


echo "======================== MFVI: BinaryMNIST + LeNet ========================="
for dssize in 10000 1000
do
    basedir="zoo/bmnist53-mfvi/BinaryMNISTC-${dssize}-53-identity/LeNet/"
    echo ">>>>>>>>>> CORRUPTION: ${CORRUPTION}    SZ = ${dssize}"
    python eval_calib.py \
        --models  ${basedir}/mfvi-sz$dssize-*\
        --corruption $CORRUPTION
    echo "-----------------------------------------------------------------"
done
echo "=========================================================================="

echo "======================== SL: BinaryMNIST + LeNet ========================="
for dssize in 10000 1000
do
    basedir="zoo/abl-alpha100-uniform-lenet/BinaryMNISTC-${dssize}-53-identity/LeNet/"
    for lam in 0.000001 0.00001 0.0001 0.001 0.01 0.1 1.0
    do
        lam_part=`printf '%1.0e' $lam`
        # model_dirs=$(ls )
        echo $model_dirs
        echo ">>>>>>>>>> CORRUPTION: ${CORRUPTION}    SZ = ${dssize}    LAM_SL = ${lam_part}"
        python eval_calib.py \
            --models  ${basedir}/sl-lam$lam_part-sz$dssize-*\
            --corruption $CORRUPTION
        echo "-----------------------------------------------------------------"
    done
done
echo "=========================================================================="


echo "======================== MFVI: BinaryMNIST + ConvNet ====================="
for dssize in 10000 1000
do
    basedir="zoo/bmnist53-mfvi/BinaryMNISTC-${dssize}-53-identity/ConvNet/"
    echo ">>>>>>>>>> CORRUPTION: ${CORRUPTION}    SZ = ${dssize}"
    python eval_calib.py \
        --models  ${basedir}/mfvi-sz$dssize-*\
        --corruption $CORRUPTION
    echo "-----------------------------------------------------------------"
done
echo "=========================================================================="

echo "====================== SL: BinaryMNIST + ConvNet ========================="
for dssize in 10000 1000
do
    basedir="zoo/abl-alpha100-uniform-convnet/BinaryMNISTC-${dssize}-53-identity/ConvNet/"
    for lam in 0.000001 0.00001 0.0001 0.001 0.01 0.1 1.0
    do
        lam_part=`printf '%1.0e' $lam`
        # model_dirs=$(ls )
        echo $model_dirs
        echo ">>>>>>>>>> CORRUPTION: ${CORRUPTION}    SZ = ${dssize}    LAM_SL = ${lam_part}"
        python eval_calib.py \
            --models  ${basedir}/sl-lam$lam_part-sz$dssize-*\
            --corruption $CORRUPTION
        echo "-----------------------------------------------------------------"
    done
done
echo "=========================================================================="
#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --exclude=dgx[1-7]
#SBATCH --output=logs/job-%A-%a.out
#SBATCH --error=logs/job-%A-%a.err

module purge
module load miniconda
source activate bayes

export DISABLE_PBAR=1


echo "======================== SLIM: BinaryMNIST + LeNet ========================="
for dssize in 10000 1000
do
    basedir="zoo/abl-a100-mnist-a050b050/BinaryMNISTC-53-identity/LeNet/"
    for lam in 0.000001 0.00001 0.0001 0.001 0.01 0.1 1.0
    do
        lam_part=`printf '%1.0e' $lam`
        echo ">>>>>>>>>> SZ = ${dssize}    LAM_SL = ${lam_part}"
        python eval_calib.py \
            --models  ${basedir}/sl-lam$lam_part-sz$dssize-*
        echo "-----------------------------------------------------------------"
    done
done
echo "=========================================================================="


echo "======================== SLIM: BinaryMNIST + ConvNet ========================="
for dssize in 10000 1000
do
    basedir="zoo/abl-a100-mnist-a050b050/BinaryMNISTC-53-identity/ConvNet/"
    for lam in 0.000001 0.00001 0.0001 0.001 0.01 0.1 1.0
    do
        lam_part=`printf '%1.0e' $lam`
        echo ">>>>>>>>>> SZ = ${dssize}    LAM_SL = ${lam_part}"
        python eval_calib.py \
            --models  ${basedir}/sl-lam$lam_part-sz$dssize-*
        echo "-----------------------------------------------------------------"
    done
done
echo "=========================================================================="

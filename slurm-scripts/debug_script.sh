#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --exclude=dgx[1-7]
#SBATCH --output=logs/job-test-%A.out
#SBATCH --error=logs/job-test-%A.err


module purge
module load miniconda
source activate bayes

export DISABLE_PBAR=1

# srun python eval_ood.py \
#     --ood FMNIST \
#     --models \
#         zoo/mfvi/BinaryMNISTC-1000-53-identity/LeNet/mfvi-sz1000-1-20220827071157 \
#         zoo/mfvi/BinaryMNISTC-1000-53-identity/LeNet/mfvi-sz1000-2-20220827075754

# srun python eval_calib.py \
#     --corruption snow \
#     --models \
#         zoo/multiclass/mfvi/CIFAR10/VGG11/mfvi-1-20220923180941 \
#         zoo/multiclass/mfvi/CIFAR10/VGG11/mfvi-1-20220926142551

# srun python eval_calib.py \
#     --corruption snow \
#     --models \
#         zoo/multiclass/edl/uniform-prior/CIFAR10/VGG11EDL/edl-1-20220927180824 \
#         zoo/multiclass/edl/uniform-prior/CIFAR10/VGG11EDL/edl-2-20220927181754

# srun python eval_ood.py \
#     --ood SVHN \
#     --models \
#         zoo/multiclass/edl/uniform-prior/CIFAR10/VGG11EDL/edl-1-20220927180824 \
#         zoo/multiclass/edl/uniform-prior/CIFAR10/VGG11EDL/edl-2-20220927181754

# # EDL
# OUTDIR="zoo/test"
# METHOD="edl"

# python train.py \
#     --method $METHOD --params annealing_step=1000 \
#     --dataset CIFAR10 --transform normalize_x_cifar_v2 \
#     --model VGG11EDL \
#     --max-steps 3000 \
#     --batch-size 256 \
#     --outdir $OUTDIR \
#     --prefix $METHOD-$SLURM_ARRAY_TASK_ID


# python train.py \
#     --method sl --params beta,a=1,b=1,alpha=1000 \
#     --dataset SSTBERT --ds-params corruption=eps-0.75 --transform normalize_x_sst \
#     --model SSTNet \
#     --max-steps 1000 \
#     --batch-size 256 \
#     --outdir zoo/test/ \
#     --prefix sst-test


# python train.py \
#     --method sl --params auto,ea=0.98,alpha=2000,adahist \
#     --dataset SSTBERT --transform normalize_x_sst \
#     --model SSTNet \
#     --max-steps 2000 \
#     --batch-size 256 \
#     --outdir zoo/test/ \
#     --prefix sst-test


# python train.py \
#     --method edl \
#     --dataset SSTBERT --transform normalize_x_sst \
#     --model SSTNetEDL \
#     --max-steps 1000 \
#     --batch-size 256 \
#     --outdir zoo/test/ \
#     --prefix sst-test


# python train.py \
#     --method sgd \
#     --dataset CIFAR10 --transform normalize_x_cifar_v2 \
#     --model VGG11Deterministic \
#     --max-steps 100 \
#     --batch-size 256 \
#     --outdir zoo/test/ \
#     --prefix sgd-test

# general config
# MAX_STEPS=5000

# python train.py \
#     --method sgd \
#     --dataset CIFAR10 --transform normalize_x_cifar_v2 \
#     --model VGG11Deterministic --model-params pretrained,head-only \
#     --max-steps $MAX_STEPS \
#     --batch-size 256 \
#     --outdir zoo/test/ \
#     --prefix sgd-prehead


# python train.py \
#     --method sgd \
#     --dataset CIFAR10 --transform cifar_da_x \
#     --model VGG11Deterministic \
#     --max-steps $MAX_STEPS \
#     --batch-size 256 \
#     --outdir zoo/test/ \
#     --prefix sgd-w-da

# python train.py \
#     --method sgdsl --params alpha=1000 \
#     --dataset CIFAR10 --transform cifar_da_x \
#     --model VGG11Deterministic \
#     --max-steps $MAX_STEPS \
#     --batch-size 256 \
#     --outdir zoo/test/ \
#     --prefix sgd-sl

# python train.py \
#     --method sgdsl --params alpha=1000 \
#     --dataset CIFAR10 --transform normalize_x_cifar_v2 \
#     --model VGG11Deterministic \
#     --max-steps $MAX_STEPS \
#     --batch-size 256 \
#     --outdir zoo/test/ \
#     --prefix sgd-sl-noaug


# OUTDIR="zoo/test/CIFAR10/VGG11Deterministic"
# CORRUPTION="identity"
# for level in {1..5}
# do
#     python eval_calib.py \
#     --corruption $CORRUPTION-$level \
#     --models ${OUTDIR}/sgd-*
# done
# python eval_calib.py \
#     --corruption identity-1 \
#     --models ${OUTDIR}/sgd-sl-noaug-*

# echo "-------------------------------------------------------------------------"

# python eval_ood.py \
#     --ood SVHN \
#     --models $OUTDIR/sgd-sl-noaug-*

# python eval_ood_save_preds.py \
#     --ood SVHN \
#     --models zoo/multiclass/mfvi/CIFAR10Im/VGG11/mfvi-*

# python eval_ood_save_preds.py \
#     --ood SVHN \
#     --models zoo/multiclass/slim/CIFAR10Im/VGG11/slim-alpha5e+02-*


# python eval_ood_save_preds.py \
#     --ood SVHN \
#     --models zoo/multiclass/edl/uniform-prior/CIFAR10Im/VGG11EDL/edl-*


python eval_ood_save_preds.py \
    --ood SVHN \
    --models zoo/multiclass/mfvi/CIFAR10/VGG11/mfvi-*

python eval_ood_save_preds.py \
    --ood SVHN \
    --models zoo/multiclass/sl/CIFAR10/VGG11/sl-alpha5e+02-*


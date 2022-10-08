#!/bin/bash
#SBATCH --time=0:30:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/debug-job-%A.out
#SBATCH --error=logs/debug-job-%A.err

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


python train.py \
    --method sl --params auto,ea=0.98,alpha=2000,adahist \
    --dataset SSTBERT --transform normalize_x_sst \
    --model SSTNet \
    --max-steps 2000 \
    --batch-size 256 \
    --outdir zoo/test/ \
    --prefix sst-test


# python train.py \
#     --method edl \
#     --dataset SSTBERT --transform normalize_x_sst \
#     --model SSTNetEDL \
#     --max-steps 1000 \
#     --batch-size 256 \
#     --outdir zoo/test/ \
#     --prefix sst-test

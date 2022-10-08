#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --exclude=dgx[1-7]
#SBATCH --output=logs/job-%A.out
#SBATCH --error=logs/job-%A.err

module purge
module load miniconda
source activate bayes

export DISABLE_PBAR=1

OOD_DATASET="FMNIST"


# # ---------------------------------- MFVI --------------------------------------
# OUTDIR="zoo/mfvi"
# METHOD="mfvi"
# for dssize in 8000 1000
# do
#     # LeNet
#     echo "--------------- MFVI: LeNet, SZ = ${dssize} ------------"
#     python eval_ood.py \
#         --ood $OOD_DATASET \
#         --models $OUTDIR/BinaryMNISTC-${dssize}-53-identity/LeNet/$METHOD-sz$dssize-*
    
#     # ConvNet
#     echo "--------------- MFVI: ConvNet, SZ = ${dssize} ------------"
#     python eval_ood.py \
#         --ood $OOD_DATASET \
#         --models $OUTDIR/BinaryMNISTC-${dssize}-53-identity/ConvNet/$METHOD-sz$dssize-*
# done


# # ----------------------------------- LS ---------------------------------------
# OUTDIR="zoo/ls"
# METHOD="ls"
# for dssize in 8000 1000
# do
#     # LeNet
#     echo "--------------- LS: LeNet, SZ = ${dssize} ------------"
#     python eval_ood.py \
#         --ood $OOD_DATASET \
#         --models $OUTDIR/BinaryMNISTC-${dssize}-53-identity/LeNet/$METHOD-sz$dssize-*
    
#     # ConvNet
#     echo "--------------- LS: ConvNet, SZ = ${dssize} ------------"
#     python eval_ood.py \
#         --ood $OOD_DATASET \
#         --models $OUTDIR/BinaryMNISTC-${dssize}-53-identity/ConvNet/$METHOD-sz$dssize-*
# done


# # ----------------------------------- EDL --------------------------------------
# OUTDIR="zoo/edl/uniform-prior"
# METHOD="edl"
# for dssize in 8000 1000
# do
#     # LeNet
#     echo "--------------- EDL: LeNet, SZ = ${dssize} ------------"
#     python eval_ood.py \
#         --ood $OOD_DATASET \
#         --models $OUTDIR/BinaryMNISTC-${dssize}-53-identity/LeNetEDL/$METHOD-sz$dssize-*
    
#     # ConvNet
#     echo "--------------- EDL: ConvNet, SZ = ${dssize} ------------"
#     python eval_ood.py \
#         --ood $OOD_DATASET \
#         --models $OUTDIR/BinaryMNISTC-${dssize}-53-identity/ConvNetEDL/$METHOD-sz$dssize-*
# done


# # ---------------------------- EDL + computed ----------------------------------
# OUTDIR="zoo/edl/computed-prior"
# METHOD="edl"
# for dssize in 8000 1000
# do
#     # LeNet
#     echo "--------------- EDL: LeNet, SZ = ${dssize} ------------"
#     python eval_ood.py \
#         --ood $OOD_DATASET \
#         --models $OUTDIR/BinaryMNISTC-${dssize}-53-identity/LeNetEDL/$METHOD-sz$dssize-*
    
#     # ConvNet
#     echo "--------------- EDL: ConvNet, SZ = ${dssize} ------------"
#     python eval_ood.py \
#         --ood $OOD_DATASET \
#         --models $OUTDIR/BinaryMNISTC-${dssize}-53-identity/ConvNetEDL/$METHOD-sz$dssize-*
# done


# # ---------------------------- EDL + 0.1 ---------------------------------------
# OUTDIR="zoo/edl/skewed-prior"
# METHOD="edl"
# for dssize in 8000 1000
# do
#     # LeNet
#     echo "--------------- EDL: LeNet, SZ = ${dssize} ------------"
#     python eval_ood.py \
#         --ood $OOD_DATASET \
#         --models $OUTDIR/BinaryMNISTC-${dssize}-53-identity/LeNetEDL/$METHOD-sz$dssize-*
    
#     # ConvNet
#     echo "--------------- EDL: ConvNet, SZ = ${dssize} ------------"
#     python eval_ood.py \
#         --ood $OOD_DATASET \
#         --models $OUTDIR/BinaryMNISTC-${dssize}-53-identity/ConvNetEDL/$METHOD-sz$dssize-*
# done


# -------------------------------- SL (auto prior) -----------------------------
OUTDIR="zoo/binary/sl-uneqbin/auto-prior-alphavar"
METHOD="sl"
for alpha in 10.0 50.0 75.0
do
    for dssize in 8000 1000
    do
        alpha_part=`printf '%1.0e' $alpha`

        # LeNet
        echo "--------------- SL: LeNet, alpha = ${alpha_part}, SZ = ${dssize} ------------"
        python eval_ood.py \
            --ood $OOD_DATASET \
            --models $OUTDIR/BinaryMNISTC-${dssize}-53-identity/LeNet/$METHOD-alpha$alpha_part-sz$dssize-*
        
        # ConvNet
        echo "------------- SL: ConvNet, alpha = ${alpha_part}, SZ = ${dssize} ------------"
        python eval_ood.py \
            --ood $OOD_DATASET \
            --models $OUTDIR/BinaryMNISTC-${dssize}-53-identity/ConvNet/$METHOD-alpha$alpha_part-sz$dssize-*

    done
done

# ------------------------------ SL (uniform prior) ----------------------------
OUTDIR="zoo/binary/sl-uneqbin/uniform-prior-alphavar"
METHOD="sl"
for alpha in 10.0 50.0 75.0
do
    for dssize in 8000 1000
    do
        alpha_part=`printf '%1.0e' $alpha`
        
        # LeNet
        echo "--------------- SL: LeNet, alpha = ${alpha_part}, SZ = ${dssize} ------------"
        python eval_ood.py \
            --ood $OOD_DATASET \
            --models $OUTDIR/BinaryMNISTC-${dssize}-53-identity/LeNet/$METHOD-alpha$alpha_part-sz$dssize-*
        
        # ConvNet
        echo "--------------- SL: ConvNet, alpha = ${alpha_part}, SZ = ${dssize} ------------"
        python eval_ood.py \
            --ood $OOD_DATASET \
            --models $OUTDIR/BinaryMNISTC-${dssize}-53-identity/ConvNet/$METHOD-alpha$alpha_part-sz$dssize-*
    done
done

# # -------------------------------- SL (0.5 prior) ------------------------------
# OUTDIR="zoo/binary/sl-uneqbin/half-prior-alphavar"
# METHOD="sl"

# for alpha in 0.01 0.1 1.0 10.0 100.0 0.05 0.5 5.0 25.0 50.0 75.0
# do
#     for dssize in 8000 1000
#     do
#         alpha_part=`printf '%1.0e' $alpha`
        
#         # LeNet
#         echo "--------------- SL: LeNet, alpha = ${alpha_part}, SZ = ${dssize} ------------"
#         python eval_ood.py \
#             --ood $OOD_DATASET \
#             --models $OUTDIR/BinaryMNISTC-${dssize}-53-identity/LeNet/$METHOD-alpha$alpha_part-sz$dssize-*
        
#         # ConvNet
#         echo "--------------- SL: ConvNet, alpha = ${alpha_part}, SZ = ${dssize} ------------"
#         python eval_ood.py \
#             --ood $OOD_DATASET \
#             --models $OUTDIR/BinaryMNISTC-${dssize}-53-identity/ConvNet/$METHOD-alpha$alpha_part-sz$dssize-*
#     done
# done


# # -------------------------------- SL (auto prior) -----------------------------
# OUTDIR="zoo/binary/sl-eqbin/auto-prior-alphavar"
# METHOD="sl"
# for alpha in 10.0 50.0 75.0
# do
#     for dssize in 8000 1000
#     do
#         alpha_part=`printf '%1.0e' $alpha`

#         # LeNet
#         echo "--------------- SL: LeNet, alpha = ${alpha_part}, SZ = ${dssize} ------------"
#         python eval_ood.py \
#             --ood $OOD_DATASET \
#             --models $OUTDIR/BinaryMNISTC-${dssize}-53-identity/LeNet/$METHOD-alpha$alpha_part-sz$dssize-*
        
#         # ConvNet
#         echo "------------- SL: ConvNet, alpha = ${alpha_part}, SZ = ${dssize} ------------"
#         python eval_ood.py \
#             --ood $OOD_DATASET \
#             --models $OUTDIR/BinaryMNISTC-${dssize}-53-identity/ConvNet/$METHOD-alpha$alpha_part-sz$dssize-*

#     done
# done

# # ------------------------------ SL (uniform prior) ----------------------------
# OUTDIR="zoo/binary/sl-eqbin/uniform-prior-alphavar"
# METHOD="sl"
# for alpha in 10.0 50.0 75.0
# do
#     for dssize in 8000 1000
#     do
#         alpha_part=`printf '%1.0e' $alpha`
        
#         # LeNet
#         echo "--------------- SL: LeNet, alpha = ${alpha_part}, SZ = ${dssize} ------------"
#         python eval_ood.py \
#             --ood $OOD_DATASET \
#             --models $OUTDIR/BinaryMNISTC-${dssize}-53-identity/LeNet/$METHOD-alpha$alpha_part-sz$dssize-*
        
#         # ConvNet
#         echo "--------------- SL: ConvNet, alpha = ${alpha_part}, SZ = ${dssize} ------------"
#         python eval_ood.py \
#             --ood $OOD_DATASET \
#             --models $OUTDIR/BinaryMNISTC-${dssize}-53-identity/ConvNet/$METHOD-alpha$alpha_part-sz$dssize-*
#     done
# done

# # -------------------------------- SL (0.5 prior) ------------------------------
# OUTDIR="zoo/binary/sl-eqbin/half-prior-alphavar"
# METHOD="sl"

# for alpha in 0.01 0.1 1.0 10.0 100.0 0.05 0.5 5.0 25.0 50.0 75.0
# do
#     for dssize in 8000 1000
#     do
#         alpha_part=`printf '%1.0e' $alpha`
        
#         # LeNet
#         echo "--------------- SL: LeNet, alpha = ${alpha_part}, SZ = ${dssize} ------------"
#         python eval_ood.py \
#             --ood $OOD_DATASET \
#             --models $OUTDIR/BinaryMNISTC-${dssize}-53-identity/LeNet/$METHOD-alpha$alpha_part-sz$dssize-*
        
#         # ConvNet
#         echo "--------------- SL: ConvNet, alpha = ${alpha_part}, SZ = ${dssize} ------------"
#         python eval_ood.py \
#             --ood $OOD_DATASET \
#             --models $OUTDIR/BinaryMNISTC-${dssize}-53-identity/ConvNet/$METHOD-alpha$alpha_part-sz$dssize-*
#     done
# done

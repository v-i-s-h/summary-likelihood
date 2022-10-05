#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --exclude=dgx[1-7]
#SBATCH --array=1-5
#SBATCH --output=logs/job-%A-%a.out
#SBATCH --error=logs/job-%A-%a.err

module purge
module load miniconda
source activate bayes

export DISABLE_PBAR=1


# # -------------------------------- SL (auto prior) -----------------------------
# OUTDIR="zoo/binary/sl-uneqbin/auto-prior-alphavar"
# MAX_STEPS=3000
# METHOD="sl"

# for alpha in 100.0 500.0 1000.0 2500.0 5000.0 7500.0 10000.0
# do
#     for dssize in 8000 1000
#     do
#         alpha_part=`printf '%1.0e' $alpha`

#         # LeNet
#         python train.py \
#             --method $METHOD --params auto,ea=0.98,adahist,alpha=$alpha \
#             --dataset BinaryMNISTC \
#             --ds-params size=$dssize,labels=53,corruption=identity \
#             --model LeNet \
#             --max-steps $MAX_STEPS \
#             --batch-size 256 \
#             --mc-samples 32 \
#             --outdir $OUTDIR \
#             --prefix $METHOD-alpha$alpha_part-sz$dssize-$SLURM_ARRAY_TASK_ID

#         # ConvNet
#         python train.py \
#             --method $METHOD --params auto,ea=0.98,alpha=$alpha \
#             --dataset BinaryMNISTC \
#             --ds-params size=$dssize,labels=53,corruption=identity \
#             --model ConvNet \
#             --max-steps $MAX_STEPS \
#             --batch-size 256 \
#             --mc-samples 32 \
#             --outdir $OUTDIR \
#             --prefix $METHOD-alpha$alpha_part-sz$dssize-$SLURM_ARRAY_TASK_ID
#     done
# done


# # ------------------------------ SL (uniform prior) ----------------------------
# OUTDIR="zoo/binary/sl-uneqbin/uniform-prior-alphavar"
# MAX_STEPS=3000
# METHOD="sl"

# for alpha in 100.0 500.0 1000.0 2500.0 5000.0 7500.0 10000.0
# do
#     for dssize in 8000 1000
#     do
#         alpha_part=`printf '%1.0e' $alpha`

#         # LeNet
#         python train.py \
#             --method $METHOD --params beta,a=1,b=1,adahist,alpha=$alpha \
#             --dataset BinaryMNISTC \
#             --ds-params size=$dssize,labels=53,corruption=identity \
#             --model LeNet \
#             --max-steps $MAX_STEPS \
#             --batch-size 256 \
#             --mc-samples 32 \
#             --outdir $OUTDIR \
#             --prefix $METHOD-alpha$alpha_part-sz$dssize-$SLURM_ARRAY_TASK_ID

#         # ConvNet
#         python train.py \
#             --method $METHOD --params beta,a=1,b=1,alpha=$alpha \
#             --dataset BinaryMNISTC \
#             --ds-params size=$dssize,labels=53,corruption=identity \
#             --model ConvNet \
#             --max-steps $MAX_STEPS \
#             --batch-size 256 \
#             --mc-samples 32 \
#             --outdir $OUTDIR \
#             --prefix $METHOD-alpha$alpha_part-sz$dssize-$SLURM_ARRAY_TASK_ID
#     done
# done


# # -------------------------------- SL (0.5 prior) ------------------------------
# OUTDIR="zoo/binary/sl-uneqbin/half-prior-alphavar"
# MAX_STEPS=3000
# METHOD="sl"

# for alpha in 100.0 500.0 1000.0 2500.0 5000.0 7500.0 10000.0
# do
#     # for dssize in 8000 1000
#     for dssize in 1000
#     do
#         alpha_part=`printf '%1.0e' $alpha`

#         # LeNet
#         python train.py \
#             --method $METHOD --params beta,a=0.5,b=0.5,adahist,alpha=$alpha \
#             --dataset BinaryMNISTC \
#             --ds-params size=$dssize,labels=53,corruption=identity \
#             --model LeNet \
#             --max-steps $MAX_STEPS \
#             --batch-size 256 \
#             --mc-samples 32 \
#             --outdir $OUTDIR \
#             --prefix $METHOD-alpha$alpha_part-sz$dssize-$SLURM_ARRAY_TASK_ID

#         # ConvNet
#         python train.py \
#             --method $METHOD --params beta,a=0.5,b=0.5,alpha=$alpha \
#             --dataset BinaryMNISTC \
#             --ds-params size=$dssize,labels=53,corruption=identity \
#             --model ConvNet \
#             --max-steps $MAX_STEPS \
#             --batch-size 256 \
#             --mc-samples 32 \
#             --outdir $OUTDIR \
#             --prefix $METHOD-alpha$alpha_part-sz$dssize-$SLURM_ARRAY_TASK_ID
#     done
# done

# -------------------------------- SL (auto prior) -----------------------------
OUTDIR="zoo/binary/sl-eqbin/auto-prior-alphavar"
MAX_STEPS=3000
METHOD="sl"

for alpha in 100.0 500.0 1000.0 2500.0 5000.0 7500.0 10000.0
do
    for dssize in 8000 1000
    do
        alpha_part=`printf '%1.0e' $alpha`

        # LeNet
        python train.py \
            --method $METHOD --params auto,ea=0.98,alpha=$alpha \
            --dataset BinaryMNISTC \
            --ds-params size=$dssize,labels=53,corruption=identity \
            --model LeNet \
            --max-steps $MAX_STEPS \
            --batch-size 256 \
            --mc-samples 32 \
            --outdir $OUTDIR \
            --prefix $METHOD-alpha$alpha_part-sz$dssize-$SLURM_ARRAY_TASK_ID

        # ConvNet
        python train.py \
            --method $METHOD --params auto,ea=0.98,alpha=$alpha \
            --dataset BinaryMNISTC \
            --ds-params size=$dssize,labels=53,corruption=identity \
            --model ConvNet \
            --max-steps $MAX_STEPS \
            --batch-size 256 \
            --mc-samples 32 \
            --outdir $OUTDIR \
            --prefix $METHOD-alpha$alpha_part-sz$dssize-$SLURM_ARRAY_TASK_ID
    done
done


# ------------------------------ SL (uniform prior) ----------------------------
OUTDIR="zoo/binary/sl-eqbin/uniform-prior-alphavar"
MAX_STEPS=3000
METHOD="sl"

for alpha in 100.0 500.0 1000.0 2500.0 5000.0 7500.0 10000.0
do
    for dssize in 8000 1000
    do
        alpha_part=`printf '%1.0e' $alpha`

        # LeNet
        python train.py \
            --method $METHOD --params beta,a=1,b=1,alpha=$alpha \
            --dataset BinaryMNISTC \
            --ds-params size=$dssize,labels=53,corruption=identity \
            --model LeNet \
            --max-steps $MAX_STEPS \
            --batch-size 256 \
            --mc-samples 32 \
            --outdir $OUTDIR \
            --prefix $METHOD-alpha$alpha_part-sz$dssize-$SLURM_ARRAY_TASK_ID

        # ConvNet
        python train.py \
            --method $METHOD --params beta,a=1,b=1,alpha=$alpha \
            --dataset BinaryMNISTC \
            --ds-params size=$dssize,labels=53,corruption=identity \
            --model ConvNet \
            --max-steps $MAX_STEPS \
            --batch-size 256 \
            --mc-samples 32 \
            --outdir $OUTDIR \
            --prefix $METHOD-alpha$alpha_part-sz$dssize-$SLURM_ARRAY_TASK_ID
    done
done


# -------------------------------- SL (0.5 prior) ------------------------------
OUTDIR="zoo/binary/sl-eqbin/half-prior-alphavar"
MAX_STEPS=3000
METHOD="sl"

for alpha in 100.0 500.0 1000.0 2500.0 5000.0 7500.0 10000.0
do
    # for dssize in 8000 1000
    for dssize in 1000
    do
        alpha_part=`printf '%1.0e' $alpha`

        # LeNet
        python train.py \
            --method $METHOD --params beta,a=0.5,b=0.5,alpha=$alpha \
            --dataset BinaryMNISTC \
            --ds-params size=$dssize,labels=53,corruption=identity \
            --model LeNet \
            --max-steps $MAX_STEPS \
            --batch-size 256 \
            --mc-samples 32 \
            --outdir $OUTDIR \
            --prefix $METHOD-alpha$alpha_part-sz$dssize-$SLURM_ARRAY_TASK_ID

        # ConvNet
        python train.py \
            --method $METHOD --params beta,a=0.5,b=0.5,alpha=$alpha \
            --dataset BinaryMNISTC \
            --ds-params size=$dssize,labels=53,corruption=identity \
            --model ConvNet \
            --max-steps $MAX_STEPS \
            --batch-size 256 \
            --mc-samples 32 \
            --outdir $OUTDIR \
            --prefix $METHOD-alpha$alpha_part-sz$dssize-$SLURM_ARRAY_TASK_ID
    done
done


# # ---------------------------------- MFVI --------------------------------------
# OUTDIR="zoo/mfvi"
# MAX_STEPS=3000
# METHOD="mfvi"
# for dssize in 8000 1000
# do
#     # LeNet
#     python train.py \
#         --method $METHOD \
#         --dataset BinaryMNISTC \
#         --ds-params size=$dssize,labels=53,corruption=identity \
#         --model LeNet \
#         --max-steps $MAX_STEPS \
#         --batch-size 256 \
#         --mc-samples 32 \
#         --outdir $OUTDIR \
#         --prefix $METHOD-sz$dssize-$SLURM_ARRAY_TASK_ID

#     # ConvNet
#     python train.py \
#         --method $METHOD \
#         --dataset BinaryMNISTC \
#         --ds-params size=$dssize,labels=53,corruption=identity \
#         --model ConvNet \
#         --max-steps $MAX_STEPS \
#         --batch-size 256 \
#         --mc-samples 32 \
#         --outdir $OUTDIR \
#         --prefix $METHOD-sz$dssize-$SLURM_ARRAY_TASK_ID
# done


# # ------------------------------------- LS -------------------------------------
# OUTDIR="zoo/ls"
# MAX_STEPS=3000
# METHOD="ls"
# for dssize in 8000 1000
# do
#     # LeNet
#     python train.py \
#         --method $METHOD \
#         --dataset BinaryMNISTC \
#         --ds-params size=$dssize,labels=53,corruption=identity \
#         --model LeNet \
#         --max-steps $MAX_STEPS \
#         --batch-size 256 \
#         --mc-samples 32 \
#         --outdir $OUTDIR \
#         --prefix $METHOD-sz$dssize-$SLURM_ARRAY_TASK_ID

#     # ConvNet
#     python train.py \
#         --method $METHOD \
#         --dataset BinaryMNISTC \
#         --ds-params size=$dssize,labels=53,corruption=identity \
#         --model ConvNet \
#         --max-steps $MAX_STEPS \
#         --batch-size 256 \
#         --mc-samples 32 \
#         --outdir $OUTDIR \
#         --prefix $METHOD-sz$dssize-$SLURM_ARRAY_TASK_ID
# done


# # ------------------------------------- EDL -------------------------------------
# OUTDIR="zoo/edl"
# MAX_STEPS=3000
# METHOD="edl"
# for dssize in 8000 1000
# do
#     # LeNet
#     python train.py \
#         --method $METHOD \
#         --dataset BinaryMNISTC \
#         --ds-params size=$dssize,labels=53,corruption=identity \
#         --model LeNetEDL \
#         --max-steps $MAX_STEPS \
#         --batch-size 256 \
#         --mc-samples 32 \
#         --outdir $OUTDIR \
#         --prefix $METHOD-sz$dssize-$SLURM_ARRAY_TASK_ID

#     # ConvNet
#     python train.py \
#         --method $METHOD \
#         --dataset BinaryMNISTC \
#         --ds-params size=$dssize,labels=53,corruption=identity \
#         --model ConvNetEDL \
#         --max-steps $MAX_STEPS \
#         --batch-size 256 \
#         --mc-samples 32 \
#         --outdir $OUTDIR \
#         --prefix $METHOD-sz$dssize-$SLURM_ARRAY_TASK_ID
# done

# # --------------------------------- EDL + computed -----------------------------
# OUTDIR="zoo/edl/computed-prior"
# MAX_STEPS=3000
# METHOD="edl"
# for dssize in 8000 1000
# do
#     # LeNet
#     python train.py \
#         --method $METHOD \
#         --params evidence_prior=computed \
#         --dataset BinaryMNISTC \
#         --ds-params size=$dssize,labels=53,corruption=identity \
#         --model LeNetEDL \
#         --max-steps $MAX_STEPS \
#         --batch-size 256 \
#         --mc-samples 32 \
#         --outdir $OUTDIR \
#         --prefix $METHOD-sz$dssize-$SLURM_ARRAY_TASK_ID

#     # ConvNet
#     python train.py \
#         --method $METHOD \
#         --params evidence_prior=computed \
#         --dataset BinaryMNISTC \
#         --ds-params size=$dssize,labels=53,corruption=identity \
#         --model ConvNetEDL \
#         --max-steps $MAX_STEPS \
#         --batch-size 256 \
#         --mc-samples 32 \
#         --outdir $OUTDIR \
#         --prefix $METHOD-sz$dssize-$SLURM_ARRAY_TASK_ID
# done

# # --------------------------------- EDL + 0.1 ----------------------------------
# OUTDIR="zoo/edl/skewed-prior"
# MAX_STEPS=3000
# METHOD="edl"
# for dssize in 8000 1000
# do
#     # LeNet
#     python train.py \
#         --method $METHOD \
#         --params evidence_prior=0.1 \
#         --dataset BinaryMNISTC \
#         --ds-params size=$dssize,labels=53,corruption=identity \
#         --model LeNetEDL \
#         --max-steps $MAX_STEPS \
#         --batch-size 256 \
#         --mc-samples 32 \
#         --outdir $OUTDIR \
#         --prefix $METHOD-sz$dssize-$SLURM_ARRAY_TASK_ID

#     # ConvNet
#     python train.py \
#         --method $METHOD \
#         --params evidence_prior=0.1 \
#         --dataset BinaryMNISTC \
#         --ds-params size=$dssize,labels=53,corruption=identity \
#         --model ConvNetEDL \
#         --max-steps $MAX_STEPS \
#         --batch-size 256 \
#         --mc-samples 32 \
#         --outdir $OUTDIR \
#         --prefix $METHOD-sz$dssize-$SLURM_ARRAY_TASK_ID
# done

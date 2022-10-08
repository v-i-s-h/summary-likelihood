#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=4
#SBATCH --array=1-5
#SBATCH --output=logs/sst-job-%A-%a.out
#SBATCH --error=logs/sst-job-%A-%a.err


module purge
module load miniconda
source activate bayes

export DISABLE_PBAR=1


MAX_STEPS=2000


# # -------------------------------- SL (auto prior) -----------------------------
# OUTDIR="zoo/sst/sl-uneqbin/auto-prior-alphavar"
# METHOD="sl"
# for alpha in 100.0 500.0 1000.0 2500.0 5000.0 10000.0
# do
#     alpha_part=`printf '%1.0e' $alpha`

#     # SSTNet
#     python train.py \
#         --method $METHOD --params auto,ea=0.98,adahist,alpha=$alpha \
#         --dataset SSTBERT --transform normalize_x_sst \
#         --model SSTNet \
#         --max-steps $MAX_STEPS \
#         --batch-size 256 \
#         --outdir $OUTDIR \
#         --prefix $METHOD-alpha$alpha_part-$SLURM_ARRAY_TASK_ID
# done

# OUTDIR="zoo/sst/sl-eqbin/auto-prior-alphavar"
# METHOD="sl"
# for alpha in 100.0 500.0 1000.0 2500.0 5000.0 10000.0
# do
#     alpha_part=`printf '%1.0e' $alpha`

#     # SSTNet
#     python train.py \
#         --method $METHOD --params auto,ea=0.98,alpha=$alpha \
#         --dataset SSTBERT --transform normalize_x_sst \
#         --model SSTNet \
#         --max-steps $MAX_STEPS \
#         --batch-size 256 \
#         --outdir $OUTDIR \
#         --prefix $METHOD-alpha$alpha_part-$SLURM_ARRAY_TASK_ID
# done


# # -------------------------------- SL (uniform prior) --------------------------
# OUTDIR="zoo/sst/sl-uneqbin/uniform-prior-alphavar"
# METHOD="sl"
# for alpha in 100.0 500.0 1000.0 2500.0 5000.0 10000.0
# do
#     alpha_part=`printf '%1.0e' $alpha`

#     # SSTNet
#     python train.py \
#         --method $METHOD --params beta,a=1,b=1,adahist,alpha=$alpha \
#         --dataset SSTBERT --transform normalize_x_sst \
#         --model SSTNet \
#         --max-steps $MAX_STEPS \
#         --batch-size 256 \
#         --outdir $OUTDIR \
#         --prefix $METHOD-alpha$alpha_part-$SLURM_ARRAY_TASK_ID
# done

# OUTDIR="zoo/sst/sl-eqbin/uniform-prior-alphavar"
# METHOD="sl"
# for alpha in 100.0 500.0 1000.0 2500.0 5000.0 10000.0
# do
#     alpha_part=`printf '%1.0e' $alpha`

#     # SSTNet
#     python train.py \
#         --method $METHOD --params beta,a=1,b=1,alpha=$alpha \
#         --dataset SSTBERT --transform normalize_x_sst \
#         --model SSTNet \
#         --max-steps $MAX_STEPS \
#         --batch-size 256 \
#         --outdir $OUTDIR \
#         --prefix $METHOD-alpha$alpha_part-$SLURM_ARRAY_TASK_ID
# done


# # ------------------------------------- MFVI -----------------------------------
# OUTDIR="zoo/sst/mfvi"
# METHOD="mfvi"
# python train.py \
#     --method $METHOD \
#     --dataset SSTBERT --transform normalize_x_sst \
#     --model SSTNet \
#     --max-steps $MAX_STEPS \
#     --batch-size 256 \
#     --outdir $OUTDIR \
#     --prefix $METHOD-$SLURM_ARRAY_TASK_ID


# # ------------------------------------- LS -------------------------------------
# OUTDIR="zoo/sst/ls"
# METHOD="ls"
# python train.py \
#     --method $METHOD --params smoothing=0.05 \
#     --dataset SSTBERT --transform normalize_x_sst \
#     --model SSTNet \
#     --max-steps $MAX_STEPS \
#     --batch-size 256 \
#     --outdir $OUTDIR \
#     --prefix $METHOD-$SLURM_ARRAY_TASK_ID


# ------------------------------------ EDL -------------------------------------
OUTDIR="zoo/sst/edl"
METHOD="edl"
python train.py \
    --method $METHOD \
    --dataset SSTBERT --transform normalize_x_sst \
    --model SSTNetEDL \
    --max-steps $MAX_STEPS \
    --batch-size 256 \
    --outdir $OUTDIR \
    --prefix $METHOD-$SLURM_ARRAY_TASK_ID


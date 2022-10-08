#!/bin/bash
#SBATCH --time=0:30:00
#SBATCH --mem=512M
#SBATCH --cpus-per-task=4
#SBATCH --array=1-11
#SBATCH --output=logs/sst-eval-job-%A-%a.out
#SBATCH --error=logs/sst-eval-job-%A-%a.err


module purge
module load miniconda
source activate bayes

export DISABLE_PBAR=1

case $SLURM_ARRAY_TASK_ID in
    1) CORRUPTION=eps-0.00 ;;
    2) CORRUPTION=eps-0.10 ;;
    3) CORRUPTION=eps-0.20 ;;
    4) CORRUPTION=eps-0.30 ;;
    5) CORRUPTION=eps-0.40 ;;
    6) CORRUPTION=eps-0.50 ;;
    7) CORRUPTION=eps-0.60 ;;
    8) CORRUPTION=eps-0.70 ;;
    9) CORRUPTION=eps-0.80 ;;
    10) CORRUPTION=eps-0.90 ;;
    11) CORRUPTION=eps-1.00;;
    *) CORRUPTION=eps-0.00 ;;
esac

# -------------------------------- SL (auto prior) -----------------------------
OUTDIR="zoo/sst/sl-uneqbin/auto-prior-alphavar"
METHOD="sl"
for alpha in 100.0 500.0 1000.0 2500.0 5000.0 10000.0
do
    alpha_part=`printf '%1.0e' $alpha`

    # SSTNet
    echo "----------- SL: SST, alpha = ${alpha_part} ----------"
    python eval_calib.py \
        --corruption $CORRUPTION \
        --models $OUTDIR/SSTBERT/SSTNet/$METHOD-alpha$alpha_part-*
done

OUTDIR="zoo/sst/sl-eqbin/auto-prior-alphavar"
METHOD="sl"
for alpha in 100.0 500.0 1000.0 2500.0 5000.0 10000.0
do
    alpha_part=`printf '%1.0e' $alpha`

    # SSTNet
    echo "----------- SL: SST, alpha = ${alpha_part} ----------"
    python eval_calib.py \
        --corruption $CORRUPTION \
        --models $OUTDIR/SSTBERT/SSTNet/$METHOD-alpha$alpha_part-*
done


# -------------------------------- SL (uniform prior) --------------------------
OUTDIR="zoo/sst/sl-uneqbin/uniform-prior-alphavar"
METHOD="sl"
for alpha in 100.0 500.0 1000.0 2500.0 5000.0 10000.0
do
    alpha_part=`printf '%1.0e' $alpha`

    # SSTNet
    echo "----------- SL: SST, alpha = ${alpha_part} ----------"
    python eval_calib.py \
        --corruption $CORRUPTION \
        --models $OUTDIR/SSTBERT/SSTNet/$METHOD-alpha$alpha_part-*
done

OUTDIR="zoo/sst/sl-eqbin/uniform-prior-alphavar"
METHOD="sl"
for alpha in 100.0 500.0 1000.0 2500.0 5000.0 10000.0
do
    alpha_part=`printf '%1.0e' $alpha`

    # SSTNet
    echo "----------- SL: SST, alpha = ${alpha_part} ----------"
    python eval_calib.py \
        --corruption $CORRUPTION \
        --models $OUTDIR/SSTBERT/SSTNet/$METHOD-alpha$alpha_part-*
done


# ---------------------------------- MFVI --------------------------------------
OUTDIR="zoo/sst/mfvi"
METHOD="mfvi"
echo "----------------------- MFVI ----------------------------"
python eval_calib.py \
    --corruption $CORRUPTION \
    --models $OUTDIR/SSTBERT/SSTNet/$METHOD-*


# ----------------------------------- LS ---------------------------------------
OUTDIR="zoo/sst/ls"
METHOD="ls"
echo "------------------------ LS -----------------------------"
python eval_calib.py \
    --corruption $CORRUPTION \
    --models $OUTDIR/SSTBERT/SSTNet/$METHOD-*

# ---------------------------------- EDL --------------------------------------
OUTDIR="zoo/sst/edl"
METHOD="edl"
echo "----------------------- EDL -----------------------------"
python eval_calib.py \
    --corruption $CORRUPTION \
    --models $OUTDIR/SSTBERT/SSTNetEDL/$METHOD-*


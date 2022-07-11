#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --exclude=dgx[1-7]
#SBATCH --output=job-%A.out
#SBATCH --error=job-%A-%a.err

module purge
module load miniconda
source activate bayes

python cifar10-vgg11-retrain-linear-ll.py


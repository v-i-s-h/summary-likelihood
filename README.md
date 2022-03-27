# Summary Prior
Incorporating summary prior information to Bayesian Deep Learning

## To start
```
git clone --recursive git@github.com:v-i-s-h/summary-prior.git
cd summary-prior
ln -s ./bayesian-torch-repo/bayesian_torch .
```

## Corrupted dataset

### Binary MNIST
![Binary MNIST 3 5](./resources/mnistc_35.png)


## Algorithms
### 1. Mean Field Variational Inference
Example: See [slurm script](./slurm-scripts/submit_mnistc_mfvi.sh).

### 2. Summary Likelihood
Exmaple: See [slurm script](./slurm-scripts/submit_mnistc_sl.sh).

## Requirements
1. Pytorch
2. Pytorch Lightning
3. Tensorboard
4. Scipy

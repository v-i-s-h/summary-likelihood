# Incorporating functional summary information in Bayesian neural networks using a Dirichlet process likelihood approach
Bayesian neural networks (BNNs) can account for both aleatoric and epistemic uncertainty. However, in BNNs the priors are often specified over the weights which rarely reflects true prior knowledge in large and complex neural network architectures. We present a simple approach to incorporate prior knowledge in BNNs based on external summary information about the predicted classification probabilities for a given dataset. The available summary information is incorporated as augmented data and modeled with a Dirichlet process, and we derive the corresponding \emph{Summary Evidence Lower BOund}. The approach is founded on Bayesian principles, and all hyperparameters have a proper probabilistic interpretation. We show how the method can inform the model about task difficulty and class imbalance. Extensive experiments show that, with negligible computational overhead, our method parallels and in many cases outperforms popular alternatives in accuracy, uncertainty calibration, and robustness against corruptions with both balanced and imbalanced data.

## To start
```
git submodule update --init
ln -s ./bayesian-torch-repo/bayesian_torch .
```
See `Requirements` section for setting up dependencies.

## Incorporating summary information
![Figure 1](./resources/fig01.png)



## Preparing data
The data for running the experiments can be downloaded by running the [download.sh](./data/download.sh) in `data/` directory. This will download the following datasets
1. MNIST-C
2. CIFAR10-C
3. SST

From root directory, run
```bash
./data/download.sh
```

Additionaly, for NLP task, you need to run [`create_sst_emb.py`](./data/create_sst_emb.py). This will download Sentence-BERT pretrained model and extact the embeddings for SST dataset. To run this,
```bash
cd data/
python create_sst_emb.py
cd ..
```

## Running experiments
> Under construction. Refer the [slurm-scripts](./slurm-scripts/) for now.

## Requirements
1. Pytorch
2. Pytorch Lightning
3. Tensorboard
4. Scipy
5. tbparse

Also, you can use the provided `environment.yaml` file to exactly reproduce the experiment environment.
# Train a Bayesian Neural Network with weight space prior using
# reparameterization


import os
import json
import argparse
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import transforms

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

import models
import datasets
import methods
from utils import parse_params_str
import config as cfg


def run_experiment(
        method,
        dataset, sublabel, corruption, imbalance, transform, 
        model_str, 
        max_steps, batch_size,
        base_params, nbins, alpha,
        wt_loss, lam_kl, lam_sl,
        mc_samples, 
        use_gpu,
        outdir):
    """
        Trains a model on dataset and returns model and logs

    Parameters
    ----------
    method : str
        Method to train BNN
    dataset : str
        Name of dataset
    sublabel : str
        Sublabels to select from dataset
    corruption : str
        Corruption label
    imbalance : float
        Class imbalance
    transform : str
        Input transformation
    model_str : str
        Model name
    max_steps : int
        Maximum number of epochs to train
    batch_size : int
        Minibatch size
    base_params : dict
        Dictionary of base measure parameters
    nbins : int
        Number of partitions for Dirichlet Prior
    alpha : float
        Concentration paramters for Dirichlet prior
    wt_loss : Bool
        Whether to use weighted loss for prediction loss
    mc_samples : int
        Number of Monte Carlo forwards for BNN
    use_gpu : Boolean
        Whether to use GPU or not

    """
    if use_gpu and torch.cuda.is_available():
        gpus = 1
    else:
        gpus = None


    x_transform = getattr(transforms, transform)() # returns a transformation object

    # Load dataset
    DatasetClass = getattr(datasets, dataset)
    trainset = DatasetClass(sublabel, corruption, 'train', 
                    imbalance=imbalance, transform=x_transform)
    testset = DatasetClass(sublabel, corruption, 'test', 
                    imbalance=imbalance, transform=x_transform)

    tr_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=testset,  batch_size=batch_size, shuffle=False)

    N = len(trainset)
    K = trainset.n_labels

    # Build model
    ModelClass = getattr(models, model_str)

    # Check for weighted loss function
    if wt_loss:
        print("INFO: Using weighted loss function.")
        w = [1.0, trainset.pos_weight]
    else:
        w = None

    # Set scaling constants if not provided
    if lam_kl is None:
        lam_kl = 1.0 / N
        print("INFO: Setting lam_kl = {}".format(lam_kl))
    if lam_sl is None:
        lam_sl = 1.0 / N
        print("INFO: Setting lam_sl = {}".format(lam_sl))

    # Prepare model to train
    model = ModelClass(K)

    # Prepare lightning model
    if method == 'mfvi':
        pl_model = methods.MFVI(model, lam_kl=lam_kl, class_weight=w, mc_samples=mc_samples)
    else:
        raise ValueError("Unknown method '{}'".format(method))

    tb_logger = pl_loggers.TensorBoardLogger(outdir)
    ckp_cb = ModelCheckpoint(outdir)

    trainer = Trainer(
        max_steps=max_steps,
        gpus=gpus,
        logger=tb_logger,
        callbacks=[ckp_cb]
    )

    trainer.fit(pl_model, tr_loader, test_loader)

    # # Build soft histogram estimator
    # hist_est = SoftHistogram(bins=nbins, min=0, max=1, sigma=cfg.HIST_SIGMA).to(device)

    # # Build prior distribution
    # prior = priors.build_prior(base_params, nbins, alpha, trainset, device)
    # s_obs = prior.base_mass


def main():
    # Timestamp for experiment
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    # Set up command line args parser
    parser = argparse.ArgumentParser()
    
    # Method related
    parser.add_argument('--method', type=str, required=True,
            help='Method for training model. (mfvi, sl)')
    
    # Dataset related
    parser.add_argument('--dataset', type=str, required=True,
            help='Datasets to train eval.')
    parser.add_argument('--sublabel', type=str, required=True,
            help='Sublabel inside dataset. Ex infiltration (ChestMNIST), 01 (MNIST)')
    parser.add_argument('--corruption', type=str, required=False,
            default='identity',
            help="Corruption to be used")
    parser.add_argument('--imbalance', type=float, required=False, default=None,
            help='Imbalance in dataset')
    parser.add_argument('--transform', type=str, required=False,
            default='normalize_x',
            help='Input transform to be applied. Defined in `transforms.py`')
    
    # Model related
    parser.add_argument('--model', type=str, required=True,
            help="Model to use. Options: LeNet")
    
    # Optimization related
    parser.add_argument('--max-steps', type=int, required=False, 
            default=cfg.MAX_TRAIN_STEPS, 
            help="Number of steps to train the model (Default: {})".format(cfg.MAX_TRAIN_STEPS))
    parser.add_argument('--batch_size', type=int, required=False, 
            default=cfg.BATCH_SIZE_TR_LOADER,
            help="Minibatch size (Default: {})".format(cfg.BATCH_SIZE_TR_LOADER))

    # Loss related
    parser.add_argument('--wt-loss',  dest='wt_loss', action='store_true',
            help='Weighted loss function')
    parser.set_defaults(wt_loss=False)
    parser.add_argument('--mc-samples', type=int, required=False, default=32,
            help="Number of MonteCarlo forwards for averaging")
    parser.add_argument('--lam-kl', type=float, default=None,
            help="Scaling for KL loss")
    parser.add_argument('--lam-sl', type=float, default=None,
            help="Scaling for Summary likelihood loss")

    # Base measure related
    parser.add_argument('--base-params', type=str, required=False, default="auto",
            help="Parameters for base measure")
    parser.add_argument('--nbins', type=int, required=False, default=10,
            help="Number of partitions for Dirichlet Prior (Default: 10)")
    parser.add_argument('--alpha', type=float, required=False, default=500.0,
            help="Concentration parameter for Dirichlet prior")
    
    # Others
    parser.add_argument('--outdir', type=str, required=False, default="zoo/",
            help="Parent output directory to save model")
    parser.add_argument('--gpu', dest='use_gpu', action='store_true')
    parser.add_argument('--no-gpu', dest='use_gpu', action='store_false')
    parser.add_argument('--prefix', type=str, required=False, default=None,
            help="Prefix for model directory")
    parser.set_defaults(use_gpu=True)
    
    args = parser.parse_args()

    method = args.method
    dataset = args.dataset
    sublabel = args.sublabel
    corruption = args.corruption
    imbalance = args.imbalance
    transform = args.transform
    model = args.model
    max_steps = args.max_steps
    batch_size = args.batch_size
    base_params = args.base_params
    alpha = args.alpha
    nbins = args.nbins
    wt_loss = args.wt_loss
    lam_kl = args.lam_kl
    lam_sl = args.lam_sl
    mc_samples = args.mc_samples
    use_gpu = args.use_gpu
    outdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            args.outdir, dataset+"-"+sublabel, model, 
                            args.prefix+"-"+timestamp if args.prefix else timestamp)

    # Parse any params
    base_params = parse_params_str(base_params)
    
    # Print experiment configuration
    print("Method           :", method)
    print("Dataset          :", dataset, sublabel, corruption, transform)
    print("Model            :", model)
    print("Max steps        :", max_steps)
    print("Batch size       :", batch_size)
    print("Base meassure    :", base_params)
    print("Paritions        :", nbins)
    print("alpha            :", alpha)
    print("Wieghted Loss    :", wt_loss)
    print("lam_kl           :", lam_kl)
    print("lam_sl           :", lam_sl)
    print("MC samples       :", mc_samples)
    print("Use GPU          :", use_gpu)
    print("Outdir           :", outdir)
    
    # Make output dir
    os.makedirs(outdir, exist_ok=True)

    # Writeout experiment configuration
    config_file_path = os.path.join(outdir, "config.json")
    with open(config_file_path, 'w') as fp:
        json.dump({
            'method': method,
            'dataset': dataset,
            'sublabel': sublabel,
            'corruption': corruption,
            'imbalance': imbalance,
            'transform': transform,
            'model': model,
            'max_steps': max_steps,
            'batch_size': batch_size,
            'base_params': base_params,
            'nbins': nbins,
            'alpha': alpha,
            'wt_loss': wt_loss,
            'lam_kl': lam_kl,
            'lam_sl': lam_sl,
            'mc_samples': mc_samples
        }, fp, indent=2)

    run_experiment(
        method,
        dataset, sublabel, corruption, imbalance, transform, 
        model, 
        max_steps, batch_size,
        base_params, nbins, alpha,
        wt_loss, lam_kl, lam_sl,
        mc_samples, 
        use_gpu,
        outdir)
    


if __name__=="__main__":
    main()

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
        method, method_params,
        dataset, ds_params, transform, 
        model_str, 
        max_steps, batch_size,
        # base_params, nbins, alpha,
        wt_loss, 
        # lam_kl, lam_sl,
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
    ds_params : dict
        kwargs for dataset class
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

    disable_pbar = int(os.environ.get('DISABLE_PBAR', 0)) == 1

    x_transform = getattr(transforms, transform)() # returns a transformation object

    # Load dataset
    DatasetClass = getattr(datasets, dataset)
    trainset = DatasetClass(**ds_params, split='train', transform=x_transform)
    testset = DatasetClass(**ds_params, split='test', transform=x_transform)

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

    # Prepare model to train
    model = ModelClass(K)

    MethodClass = getattr(methods, method)

    # Get default params
    method_params = MethodClass.populate_missing_params(method_params, trainset)
    pl_model = MethodClass(model, **method_params, class_weight=w, mc_samples=mc_samples)

    print(pl_model)
    
    tb_logger = pl_loggers.TensorBoardLogger(outdir)
    ckp_cb = ModelCheckpoint(outdir)

    trainer = Trainer(
        max_steps=max_steps,
        gpus=gpus,
        logger=tb_logger,
        callbacks=[ckp_cb],
        enable_progress_bar=not disable_pbar,
        log_every_n_steps=10
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
    parser.add_argument('--ds-params', type=str, required=True,
            help="Additional params for dataset in param1=val1,param2=val2,... format")
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
    parser.add_argument('--batch-size', type=int, required=False, 
            default=cfg.BATCH_SIZE_TR_LOADER,
            help="Minibatch size (Default: {})".format(cfg.BATCH_SIZE_TR_LOADER))

    # Loss related
    parser.add_argument('--wt-loss',  dest='wt_loss', action='store_true',
            help='Weighted loss function')
    parser.set_defaults(wt_loss=False)
    parser.add_argument('--mc-samples', type=int, required=False, default=32,
            help="Number of MonteCarlo forwards for averaging")
    
    # Method specific params
    # parser.add_argument('--lam-kl', type=float, default=None,
    #         help="Scaling for KL loss")
    # parser.add_argument('--lam-sl', type=float, default=None,
    #         help="Scaling for Summary likelihood loss")
    # parser.add_argument('--base-params', type=str, required=False, default="auto,ea=0.95",
    #         help="Parameters for base measure")
    # parser.add_argument('--nbins', type=int, required=False, default=10,
    #         help="Number of partitions for Dirichlet Prior (Default: 10)")
    # parser.add_argument('--alpha', type=float, required=False, default=500.0,
    #         help="Concentration parameter for Dirichlet prior")
    parser.add_argument('--params', type=str, required=False,
            help="Additional arguments for training algorithm")
    
    # Others
    parser.add_argument('--outdir', type=str, required=False, default="zoo/test/",
            help="Parent output directory to save model")
    parser.add_argument('--gpu', dest='use_gpu', action='store_true')
    parser.add_argument('--no-gpu', dest='use_gpu', action='store_false')
    parser.add_argument('--prefix', type=str, required=False, default=None,
            help="Prefix for model directory")
    parser.set_defaults(use_gpu=True)
    
    args = parser.parse_args()

    method = args.method
    method_params = args.params
    dataset = args.dataset
    ds_params = args.ds_params
    transform = args.transform
    model = args.model
    max_steps = args.max_steps
    batch_size = args.batch_size
    # base_params = args.base_params
    # alpha = args.alpha
    # nbins = args.nbins
    wt_loss = args.wt_loss
    # lam_kl = args.lam_kl
    # lam_sl = args.lam_sl
    mc_samples = args.mc_samples
    use_gpu = args.use_gpu

    # Parse any params
    ds_params = parse_params_str(ds_params)
    # base_params = parse_params_str(base_params)
    method_params = parse_params_str(method_params)

    dataset_str = dataset + '-' + '-'.join([str(v) for _, v in ds_params.items()])

    outdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            args.outdir, 
                            dataset_str, 
                            model, 
                            args.prefix+"-"+timestamp if args.prefix else timestamp)

    
    # Print experiment configuration
    print("Method           :", method, method_params)
    print("Dataset          :", dataset, ds_params, transform)
    print("Model            :", model)
    print("Max steps        :", max_steps)
    print("Batch size       :", batch_size)
    # print("Base meassure    :", base_params)
    # print("Paritions        :", nbins)
    # print("alpha            :", alpha)
    print("Weighted Loss    :", wt_loss)
    # print("lam_kl           :", lam_kl)
    # print("lam_sl           :", lam_sl)
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
            'method_params': method_params,
            'dataset': dataset,
            'ds_params': ds_params,
            'transform': transform,
            'model': model,
            'max_steps': max_steps,
            'batch_size': batch_size,
            # 'base_params': base_params,
            # 'nbins': nbins,
            # 'alpha': alpha,
            'wt_loss': wt_loss,
            # 'lam_kl': lam_kl,
            # 'lam_sl': lam_sl,
            'mc_samples': mc_samples
        }, fp, indent=2)

    run_experiment(
        method, method_params,
        dataset, ds_params, transform, 
        model, 
        max_steps, batch_size,
        # base_params, nbins, alpha,
        wt_loss, 
        # lam_kl, lam_sl,
        mc_samples, 
        use_gpu,
        outdir)
    


if __name__=="__main__":
    main()
# Evaluate model(s) on given dataset

import os
import pickle
import json
import glob
import argparse
import copy
from datetime import datetime

from tqdm import tqdm

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchmetrics.functional import calibration_error, accuracy

import pandas as pd

import models
import datasets
import transforms


class TScaling():
    def __init__(self, device='cpu'):
        self.device = device
        self.T = nn.Parameter((1.0 * torch.ones(1)).to(device))

    def T_scaling(self, logits):
        return torch.div(logits, self.T)

    def fit(self, logits, y_true, sample_weight=None):
        optimizer = optim.LBFGS([self.T], lr=1e-2, max_iter=100, line_search_fn='strong_wolfe')

        # ece_pre = F.nll_loss(torch.log(self.predict_proba(logits)), y_true).detach().item()

        def _eval():
            loss = F.cross_entropy(self.T_scaling(logits), y_true)
            loss.backward()
            # print("     ", loss.item(), self.T.item())
            return loss
        
        optimizer.step(_eval)

        # Debug
        # ece_post = F.nll_loss(torch.log(self.predict_proba(logits)), y_true).detach().item()

        # print("INFO: T    = ", self.T.item())
        # print("INFO: pre  = {:.5f}".format(ece_pre))
        # print("INFO: post = {:.5f}".format(ece_post))


    def predict_proba(self, logits):
        calib_logits = None
        with torch.no_grad():
            calib_logits = self.T_scaling(logits)
            y_pred = torch.sigmoid(calib_logits)

        y_pred = y_pred.detach()

        return y_pred


def get_predictions(model, dataloader, msg="Getting predictions"):
    device = next(model.parameters()).device
    tqdm_params = {
        'leave': False,
        'desc': msg,
        'disable': True if os.environ.get('SLURM_JOB_ID', False) else False
    }

    y_true = None
    scores = None
    logits = None

    model.eval()
    with torch.no_grad():
        for x, y in tqdm(dataloader, **tqdm_params):
            x = x.to(device)
            y = y.to(device)

            logits_ = []
            for mc_run in range(32): # TODO: Adjustable MC sampling
                mc_logits, _ = model(x)
                logits_.append(mc_logits)

            _logits = torch.mean(torch.stack(logits_), dim=0)
            score = torch.sigmoid(_logits)

            if y_true is None:
                y_true = y.detach()
            else:
                y_true = torch.concat([y_true, y.detach()])
            if logits is None:
                logits = _logits.detach()
            else:
                logits = torch.concat([logits, _logits.detach()])
            if scores is None:
                scores = score.detach()
            else:
                scores = torch.concat([scores, score.detach()])
    model.train()

    return y_true, scores, logits


def get_model_predictions(model, valloader, testloader, n_bins=10):
    # Get predictions from validation set and calibrate models
    y_true_val, scores_val, logits_val = get_predictions(model, valloader)
    
    ### Evaluate calibration of testloader
    y_true_test, scores_test, logits_test = get_predictions(model, testloader)

    return {
        'val': [ y_true_val, scores_val, logits_val],
        'test': [ y_true_test, scores_test, logits_test]
    }


def evaluate_model_calibration(preds, n_bins=10):
    # To store results
    results = {}
    
    # Get predictions from validation set and calibrate models
    y_true_val, scores_val, logits_val = preds['val']
    
    ### Evaluate calibration of testloader
    y_true_test, scores_test, logits_test = preds['test']
    
    results['nll_uncal_val'] = F.nll_loss(torch.log(scores_val), y_true_val).detach().item()
    results['nll_uncal_test'] = F.nll_loss(torch.log(scores_test), y_true_test).detach().item()

    calib_results_un = calibration_error(scores_test, y_true_test, n_bins=n_bins, norm='l1')
    results['ece_uncal'] = calib_results_un.item()
    
    # T scaling (on logits)
    tscale = TScaling(device=logits_val.device)
    tscale.fit(logits_val, y_true_val)
    scores_tscale = tscale.predict_proba(logits_test)
    
    # Evalute calibration results
    results['nll_cal_test'] = F.nll_loss(
        torch.log(scores_tscale),
        y_true_test
    ).detach().item()
    results['nll_cal_val'] = F.nll_loss(
        torch.log(tscale.predict_proba(logits_val)),
        y_true_val
    ).detach().item()

    calib_results_tscale = calibration_error(scores_tscale, y_true_test, n_bins=n_bins, norm='l1')
    results['ece_tscale'] = calib_results_tscale.item()
    results['T'] = tscale.T.item()

    results['ece_gain'] = results['nll_uncal_test'] - results['nll_cal_test']

    results['acc'] = accuracy(scores_tscale, y_true_test).detach().item()

    return results


def run_evaluation(model_str, ckpt_file, dataset, ds_params, transform, corruption):
    # Setup input transform
    x_transform = getattr(transforms, transform)() # returns a transformation object

    # Load dataset
    DatasetClass = getattr(datasets, dataset)

    # Remove size constraint on test
    if 'size' in ds_params:
        del ds_params['size']
    if corruption is not None:
        test_params = copy.deepcopy(ds_params)
        test_params.update({'corruption': corruption})
        
        # If testing on a corruption, use entire test set as validation for calibration
        print("INFO: Testing on corruption. Using entire test set of clean data for calibration")
        valset = DatasetClass(**ds_params, split='test', transform=x_transform)
        valloader = DataLoader(dataset=valset, batch_size=256, shuffle=True)

        testset = DatasetClass(**test_params, split='test', transform=x_transform)
        testloader = DataLoader(dataset=testset, batch_size=256, shuffle=False)
    else:
        print("INFO: Using part of test set for calibration")
        testset = DatasetClass(**ds_params, split='test', transform=x_transform)

        n = len(testset)
        indices = torch.randperm(n)
        val_size = int(0.25 * n)
        val_indices = indices[:val_size]
        test_indices = indices[val_size:]

        valloader = DataLoader(dataset=testset, batch_size=256, 
                                sampler=SubsetRandomSampler(val_indices))
        testloader = DataLoader(dataset=testset, batch_size=256, 
                                sampler=SubsetRandomSampler(test_indices))
    

    # CUDA config
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    
    # Build model
    ModelClass = getattr(models, model_str+'Logits')
    model = ModelClass(testset.n_labels).to(device)
    checkpoint = torch.load(ckpt_file, map_location=torch.device(device))
    # Clean up state dict to have only model params
    state_dict = checkpoint['state_dict']
    model_weights = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            # remove `model.` from start and add to state dict
            model_weights[k[6:]] = v
    model.load_state_dict(model_weights)
    # model.load_state_dict(torch.load(weights_file, map_location=torch.device(device)))

    # Get predictions from model
    preds = get_model_predictions(model, valloader, testloader)
    del model # Free model


    r = evaluate_model_calibration(preds, n_bins=10)

    return r


def evaluate_model_in_dir(model_dir, corruption):
    config_path = os.path.join(model_dir, "config.json")
    weights_path = os.path.join(model_dir, "model.pth")

    # Load experiment configuration
    with open(config_path) as fp:
        config = json.load(fp)

    # Get the best weights file
    ckpt_path = glob.glob(model_dir + '/step=*.ckpt')[-1]

    
    r = run_evaluation(
            config['model'], 
            ckpt_path, 
            config['dataset'],
            config['ds_params'], 
            config['transform'],
            corruption)

    return r  


def main():
    # Timestamp for experiment
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    # Set up command line args parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--corruption', type=str, required=False, default=None)
    parser.add_argument('--models', nargs='+', required=False)
    args = parser.parse_args()

    model_dirs = args.models
    if not isinstance(model_dirs, list):
        model_dirs = list(model_dirs)

    results = []

    for i, model_dir in tqdm(enumerate(model_dirs), desc="Models", total=len(model_dirs)):
        model_results = []
        for j in range(5):
            r = {'model_id': model_dir, 'round': j}
            _r = evaluate_model_in_dir(model_dir, args.corruption)
            r.update(_r)
            model_results.append(r)
        # Print results
        # print("Model:", model_dir)
        # print("    Uncalibrated ECE   : {:.4f}".format(r['uncal']))
        # print("    ECE after TScaling : {:.4f}".format(r['tscale']))
        # print("                     T : {:.4f}".format(r['T']))
        if args.corruption is not None:
            results_file = os.path.join(model_dir, "calib_results_{}.pkl".format(args.corruption))
        else:
            results_file = os.path.join(model_dir, "calib_results.pkl")
        with open(results_file, 'wb') as f:
            pickle.dump(model_results, f)
        results.extend(model_results)

    df_results = pd.DataFrame(results)
    # Select results where ECE is improved, others are mostly optimization failures
    df_good = df_results[df_results.ece_gain >= 0.0] 
    print(df_results)
    print("---------------------")
    print(df_good)
    print("---------------------")
    # Compute statistics
    n = df_good.shape[0]
    mean_uncalib_ece = np.mean(df_good.ece_uncal)
    mean_calib_ece = np.mean(df_good.ece_tscale)
    mean_acc = np.mean(df_good.acc)
    err_uncalib_ece = np.std(df_good.ece_uncal) / np.sqrt(n)
    err_calib_ece = np.std(df_good.ece_tscale) / np.sqrt(n)
    err_acc = np.std(df_good.acc) / np.sqrt(n)

    print("ECE (Uncalibrated) = {:.4f} \pm {:.4f}".format(mean_uncalib_ece, err_uncalib_ece))
    print("ECE (T-Scaling)    = {:.4f} \pm {:.4f}".format(mean_calib_ece, err_calib_ece))
    print("Accuracy           = {:.4f} \pm {:.4f}".format(mean_acc, err_acc))


if __name__ == "__main__":
    main()
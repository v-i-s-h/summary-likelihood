# Evaluate model(s) on given dataset

import os
import sys
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

import models
import datasets
import transforms


class TScaling():
    def __init__(self, device='cpu'):
        self.device = device
        self.T = nn.Parameter(torch.ones(1).to(device))

    def T_scaling(self, logits):
        return torch.div(logits, self.T)

    def fit(self, logits, y_true, sample_weight=None):
        optimizer = optim.LBFGS([self.T], lr=1e-3, max_iter=10000, line_search_fn='strong_wolfe')

        def _eval():
            loss = F.nll_loss(self.T_scaling(logits), y_true)
            loss.backward()

            return loss
        
        optimizer.step(_eval)

        print("End of fit. T = ", self.T)

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


def evaluate_model(model, valloader, testloader, n_bins=10):
    # To store results
    results = {}
    
    # Get predictions from validation set and calibrate models
    y_true_val, scores_val, logits_val = get_predictions(model, valloader)
    
    ### Evaluate calibration of testloader
    y_true_test, scores_test, logits_test = get_predictions(model, testloader)
    
    calib_results_un = calibration_error(scores_test, y_true_test, n_bins=n_bins)
    # acc = accuracy(scores_test, y_true_test)
    # print("Acc =", acc)
    results['uncalibrated'] = calib_results_un.item()

    # T scaling (on logits)
    tscale = TScaling(device=logits_val.device)
    tscale.fit(logits_val, y_true_val)
    scores_tscale = tscale.predict_proba(logits_test)
    calib_results_tscale = calibration_error(scores_tscale, y_true_test, n_bins=n_bins)
    results['tscale'] = calib_results_tscale.item()

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
        valloader = DataLoader(dataset=valset, batch_size=64, shuffle=True)

        testset = DatasetClass(**test_params, split='test', transform=x_transform)
        testloader = DataLoader(dataset=testset, batch_size=64, shuffle=False)
    else:
        print("INFO: Using part of test set for calibration")
        testset = DatasetClass(**ds_params, split='test', transform=x_transform)

        n = len(testset)
        indices = torch.randperm(n)
        val_size = int(0.25 * n)
        val_indices = indices[:val_size]
        test_indices = indices[val_size:]

        valloader = DataLoader(dataset=testset, batch_size=64, 
                                sampler=SubsetRandomSampler(val_indices))
        testloader = DataLoader(dataset=testset, batch_size=64, 
                                sampler=SubsetRandomSampler(test_indices))
    

    # CUDA config
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    
    # Build model
    ModelClass = getattr(models, model_str+'Logits')
    model = ModelClass(testset.n_labels).to(device)
    checkpoint = torch.load(ckpt_file)
    # Clean up state dict to have only model params
    state_dict = checkpoint['state_dict']
    model_weights = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            # remove `model.` from start and add to state dict
            model_weights[k[6:]] = v
    model.load_state_dict(model_weights)
    # model.load_state_dict(torch.load(weights_file, map_location=torch.device(device)))

    print(model)

    # Evaluate model
    r = evaluate_model(model, valloader, testloader, n_bins=10)

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
        results_file = os.path.join(model_dir, "eval-results.pkl")
        
        r = evaluate_model_in_dir(model_dir, args.corruption)
        results.append(r)
        print(r)

    #     with open(results_file, 'wb') as fp:
    #         pickle.dump(r, fp)

    #     # Plot and save the results
    #     fig_file = os.path.join(model_dir, "results.png")
    #     fig = make_results_fig(r)
    #     plt.savefig(fig_file, bbox_inches='tight')

    # # Printout comparison table
    # print("{:2s} {:<60s}    {:^6s}    {:^6s}    {:^6s}    {:^6s}".format(
    #     "", "Model", "Acc.", "F1", "AUC", "ECE"))
    # for i, (m, r) in enumerate(zip(model_dirs, results)):
    #     print("{:2d} {:<60s}    {:.4f}    {:.4f}    {:.4f}    {:.4f}".format(
    #         i+1, m, r['acc'], r['f1_score'], r['auc']['auc'], r['calib']['ece']
    #     ))

    # if len(results) > 1:
    #     # For more than one results, also produce the summary
    #     acc_list = [r['acc'] for r in results]
    #     f1_list = [r['f1_score'] for r in results]
    #     auc_list = [r['auc']['auc'] for r in results]
    #     ece_list = [r['calib']['ece'] for r in results]

    #     acc_mean = np.mean(acc_list)
    #     acc_std  = np.std(acc_list)

    #     f1_mean = np.mean(f1_list)
    #     f1_std = np.std(f1_list)

    #     auc_mean = np.mean(auc_list)
    #     auc_std = np.std(auc_list)

    #     ece_mean = np.mean(ece_list)
    #     ece_std = np.std(ece_list)

    #     print("{:2s} {:<60s}    {:^6s}    {:^6s}    {:^6s}    {:^6s}".format(
    #         "--", "-"*60, "-"*6, "-"*6, "-"*6, "-"*6
    #     ))
    #     print("{:2s} {:<60s}    {:.4f}    {:.4f}    {:.4f}    {:.4f}".format(
    #         "", "", acc_mean, f1_mean, auc_mean, ece_mean
    #     ))
    #     print("{:2s} {:<60s}   ±{:.4f}   ±{:.4f}   ±{:.4f}   ±{:.4f}".format(
    #         "", "", acc_std, f1_std, auc_std, ece_std
    #     ))


if __name__ == "__main__":
    main()
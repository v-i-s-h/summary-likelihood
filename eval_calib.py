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
from torchmetrics.functional import calibration_error, accuracy, auroc

import pandas as pd

import models
import datasets
import transforms


def get_predictions(model, dataloader, msg="Getting predictions"):
    device = next(model.parameters()).device
    tqdm_params = {
        'leave': False,
        'desc': msg,
        'disable': True if os.environ.get('SLURM_JOB_ID', False) else False
    }

    y_true = None
    scores = None
    
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(dataloader, **tqdm_params):
            x = x.to(device)
            y = y.to(device)

            ypred = []
            for mc_run in range(128): # TODO: Adjustable MC sampling
                _ypred = model.get_softmax(x)
                ypred.append(_ypred)

            score = torch.mean(torch.stack(ypred), dim=0)
            
            if y_true is None:
                y_true = y.detach()
            else:
                y_true = torch.concat([y_true, y.detach()])
            if scores is None:
                scores = score.detach()
            else:
                scores = torch.concat([scores, score.detach()])
    model.train()

    return y_true, scores


def get_model_predictions(model, valloader, testloader, n_bins=10):
    # Get predictions from validation set and calibrate models
    y_true_val, scores_val = get_predictions(model, valloader)
    
    ### Evaluate calibration of testloader
    y_true_test, scores_test = get_predictions(model, testloader)

    return {
        'val': [ y_true_val, scores_val],
        'test': [ y_true_test, scores_test]
    }


def evaluate_model_calibration(preds, n_bins=10):
    # To store results
    results = {}
    
    # Get predictions from validation set and calibrate models
    y_true_val, scores_val = preds['val']
    
    ### Evaluate calibration of testloader
    y_true_test, scores_test = preds['test']
    
    results['nll_uncal_val'] = F.nll_loss(torch.log(scores_val), y_true_val).detach().item()
    results['nll_uncal_test'] = F.nll_loss(torch.log(scores_test), y_true_test).detach().item()

    results['ece_uncal_val'] = calibration_error(
                                    scores_val, y_true_val, 
                                    n_bins=n_bins, norm='l1'
                                ).item()
    results['ece_uncal_test'] = calibration_error(
                                    scores_test, y_true_test, 
                                    n_bins=n_bins, norm='l1'
                                ).item()
    
    results['acc_val'] = accuracy(scores_val, y_true_val).detach().item()
    results['acc_test'] = accuracy(scores_test, y_true_test).detach().item()

    results['auroc_val'] = auroc(scores_val, y_true_val,
                                num_classes=preds['n_classes']).detach().item()
    results['auroc_test'] = auroc(scores_test, y_true_test,
                                num_classes=preds['n_classes']).detach().item()

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
        print("INFO: Testing on corruption {}. "
            "Using entire test set of clean data for validation".format(corruption))
        valset = DatasetClass(**ds_params, split='val', transform=x_transform)
        valloader = DataLoader(dataset=valset, batch_size=256, shuffle=True)

        testset = DatasetClass(**test_params, split='test', transform=x_transform)
        testloader = DataLoader(dataset=testset, batch_size=256, shuffle=False)
    else:
        print("INFO: Using part of test set for validation")
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
    ModelClass = getattr(models, model_str)
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
    preds['n_classes'] = testset.n_labels


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
        print("INFO: Model -", model_dir)
        model_results = []
        for j in range(1):
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
            results_file = os.path.join(model_dir, "ece_results_{}.pkl".format(args.corruption))
        else:
            results_file = os.path.join(model_dir, "ece_results.pkl")
        with open(results_file, 'wb') as f:
            pickle.dump(model_results, f)
        results.extend(model_results)

    df_results = pd.DataFrame(results)
    # Select results where ECE is improved, others are mostly optimization failures
    df_good = df_results
    print(df_good)
    print("---------------------")
    # Compute statistics
    n = df_good.shape[0]
    mean_uncalib_ece = np.mean(df_good.ece_uncal_val)
    mean_acc = np.mean(df_good.acc_val)
    err_uncalib_ece = np.std(df_good.ece_uncal_val) / np.sqrt(n)
    err_acc = np.std(df_good.acc_val) / np.sqrt(n)

    print("ECE (Uncalibrated) = {:.4f} \pm {:.4f}".format(mean_uncalib_ece, err_uncalib_ece))
    print("Accuracy           = {:.4f} \pm {:.4f}".format(mean_acc, err_acc))


if __name__ == "__main__":
    main()
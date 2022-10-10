# Evaluate model(s) on given OOD dataset

import os
import pickle
import json
import glob
import argparse
from datetime import datetime

from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

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


def get_model_predictions(model, data_loaders):
    # data_loaders is a dict of {"label": <DataLoader>}
    
    predictions = {}
    ### Get predictions on dataloaders
    for label, dataloader in tqdm(data_loaders.items(), desc="Testing datasets"):
        y_true, scores = get_predictions(model, dataloader)
        predictions[label] = [y_true, scores]

    return predictions


def compute_mean_entropy(p):
    sample_ent = torch.nansum(-torch.log(p) * p, dim=1)
    return sample_ent.nanmean(), sample_ent


def evaluate_model_performance(preds, n_bins=10):
    # To store results
    results = {}
    
    # Get the predictions for OOD dataset
    _, scores_ood = preds['ood']
    
    # Get the predictions for in domain test dataset
    _, scores_test = preds['indomain']
    
    mean_ent, sample_ent = compute_mean_entropy(scores_ood)
    results['ent_ood'] = mean_ent.detach().item()
    results['ent_ood_samples'] = sample_ent.cpu().detach().numpy()
    mean_ent, sample_ent = compute_mean_entropy(scores_test)
    results['ent_test'] = mean_ent.detach().item()
    results['ent_test_samples'] = sample_ent.cpu().detach().numpy()

    results['ent_delta'] = results['ent_ood'] - results['ent_test']

    return results


def run_evaluation(model_str, ckpt_file, dataset, ds_params, transform, ood_dataset):
    
    # Setup input transform
    x_transform = getattr(transforms, transform)() # returns a transformation object

    # Load OOD dataset
    OODDatasetClass = getattr(datasets, ood_dataset)
    ood_testset = OODDatasetClass(split='test', transform=x_transform)
    ood_testloader = DataLoader(dataset=ood_testset, batch_size=256, shuffle=False)

    # Load In-domain dataser
    DatasetClass = getattr(datasets, dataset)
    if 'size' in ds_params:
        del ds_params['size']
    testset = DatasetClass(**ds_params, split='test', transform=x_transform)
    testloader = DataLoader(dataset=testset, batch_size=256, shuffle=False)

    # CUDA config
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    
    # Load saved weights
    checkpoint = torch.load(ckpt_file, map_location=torch.device(device))
    # Clean up state dict to have only model params
    state_dict = checkpoint['state_dict']
    model_weights = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            # remove `model.` from start and add to state dict
            model_weights[k[6:]] = v

    # Get last layer's bias parameter to identify the number of labels
    # the model is trained for
    n_labels = model_weights[list(model_weights.keys())[-1]].shape[0]

    # Build model
    ModelClass = getattr(models, model_str)
    model = ModelClass(n_labels).to(device)
    model.load_state_dict(model_weights)

    # Get predictions from model
    preds = get_model_predictions(model, {
                "ood": ood_testloader,
                "indomain": testloader
            })
    del model # Free model
    preds['n_classes'] = n_labels

    r = evaluate_model_performance(preds)

    return r


def evaluate_model_in_dir(model_dir, ood_dataset):
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
            ood_dataset)

    return r  


def main():
    # Timestamp for experiment
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    # Set up command line args parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--ood', type=str, required=True, default=None)
    parser.add_argument('--models', nargs='+', required=True)
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
            _r = evaluate_model_in_dir(model_dir, args.ood)
            r.update(_r)
            model_results.append(r)
        
        results_file = os.path.join(model_dir, 
                            "ood_results_{}.pkl".format(args.ood))
        with open(results_file, 'wb') as f:
            pickle.dump(model_results, f)
        results.extend(model_results)

    df_results = pd.DataFrame(results)
    print(df_results)
    

if __name__ == "__main__":
    main()
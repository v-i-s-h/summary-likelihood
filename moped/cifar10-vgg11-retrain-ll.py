# Script for training CIFAR10 VGG16 using MOPED and then with log loss alone for calibration

import sys
sys.path.append("./../")

import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR10
from sklearn.metrics import log_loss, brier_score_loss
import sklearn.metrics as metrics

from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss

from pretrained.CIFAR10.PyTorch_CIFAR10.cifar10_models import vgg
from pretrained.MedMNIST.evaluation import ECE, MCE


def softmax(x):
    """
    Compute softmax values for each sets of scores in x.
    
    Parameters:
        x (numpy.ndarray): array containing m samples with n-dimensions (m,n)
    Returns:
        x_softmax (numpy.ndarray) softmaxed values for initial (m,n) array
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=1)


def evaluate(probs, y_true, verbose = False, normalize = False, bins = 15):
    """
    Evaluate model using various scoring measures: Error Rate, ECE, MCE, NLL, Brier Score
    
    Params:
        probs: a list containing probabilities for all the classes with a shape of (samples, classes)
        y_true: a list containing the actual class labels
        verbose: (bool) are the scores printed out. (default = False)
        normalize: (bool) in case of 1-vs-K calibration, the probabilities need to be normalized.
        bins: (int) - into how many bins are probabilities divided (default = 15)
        
    Returns:
        (error, ece, mce, loss, brier), returns various scoring measures
    """
    
    preds = np.argmax(probs, axis=1)  # Take maximum confidence as prediction
    
    if normalize:
        confs = np.max(probs, axis=1)/np.sum(probs, axis=1)
        # Check if everything below or equal to 1?
    else:
        confs = np.max(probs, axis=1)  # Take only maximum confidence
    
    accuracy = metrics.accuracy_score(y_true, preds) * 100
    error = 100 - accuracy
    
        # Calculate ECE
    ece = ECE(confs, preds, y_true, bin_size = 1/bins)
    # Calculate MCE
    mce = MCE(confs, preds, y_true, bin_size = 1/bins)
    
    loss = log_loss(y_true=y_true, y_pred=probs)
    
    y_prob_true = np.array([probs[i, idx] for i, idx in enumerate(y_true)])  # Probability of positive class
#     brier = brier_score_loss(y_true=y_true, y_prob=y_prob_true)  # Brier Score (MSE)
    
    if verbose:
        print("Accuracy:", accuracy)
        print("Error:", error)
        print("ECE:", ece)
        print("MCE:", mce)
        print("Loss:", loss)
#         print("brier:", brier)
    
    return (error, ece, mce, loss)


def get_model_predictions(model, dataloader, device):
    model.eval()

    logits = torch.tensor([])
    targets = torch.tensor([])
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            outputs = model(inputs)

            logits = torch.cat((logits, outputs.cpu()), 0)
            targets = torch.cat((targets, labels), 0)

    logits = logits.numpy()
    labels = targets.numpy().reshape(-1, 1).astype(int)

    return logits, labels


def get_bnn_model_predictions(model, dataloader, device):
    mc_samples = 32
    output_list = []
    labels_list = []
    
    model.eval()
    with torch.no_grad():
        for data, target in dataloader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            else:
                data, target = data.cpu(), target.cpu()
            output_mc = []
            for mc_run in range(mc_samples):
                output = model.forward(data)
                output_mc.append(output)
            output_ = torch.stack(output_mc).mean(dim=0)
            output_list.append(output_)
            labels_list.append(target)

        output = torch.cat(output_list)
        labels = torch.cat(labels_list)

    logits = output.cpu().numpy()
    labels = labels.cpu().numpy().reshape(-1, 1).astype(int)

    return logits, labels


def train(model, train_loader, val_loader):
    mc_samples = 32

    optimizer = torch.optim.Adam(model.parameters(), 1e-5)
    criterion = nn.CrossEntropyLoss().cuda()

    # switch to train mode
    model.train()

    val_log = {
        'err': [],
        'ece': [],
        'mce': [],
        'loss': []
    }

    for epoch_idx in range(25):
        loss_val = 0.0
        for i, (input, target) in enumerate(train_loader):
            if torch.cuda.is_available():
                target = target.cuda()
                input_var = input.cuda()
                target_var = target
            else:
                target = target.cpu()
                input_var = input.cpu()
                target_var = target

            
            # compute output
            output_ = []
            kl_ = []
            for mc_run in range(mc_samples):
                output = model(input_var)
                kl = get_kl_loss(model)
                output_.append(output)
                kl_.append(kl)
            output = torch.mean(torch.stack(output_), dim=0)
            kl = torch.mean(torch.stack(kl_), dim=0)
            cross_entropy_loss = criterion(output, target_var)
            scaled_kl = kl / 128

            # ELBO loss
            loss = cross_entropy_loss + scaled_kl

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val += loss.item()

        logits, labels = get_model_predictions(model, val_loader, 'cuda')
        val_err, val_ece, val_mce, val_loss = evaluate(softmax(logits), labels, verbose=False)  

        val_log['err'].append(val_err)
        val_log['ece'].append(val_ece)
        val_log['mce'].append(val_mce)
        val_log['loss'].append(val_loss)

        print("{:3d}    loss = {:.3f}    ece = {:.3f}    mce = {:.3f}    err = {:.3f}".format(epoch_idx,
            val_loss, val_ece, val_mce, val_err))

    return val_log


def retrain(model, train_loader, val_loader):
    mc_samples = 32

    optimizer = torch.optim.Adam(model.parameters(), 1e-5)
    # optimizer = torch.optim.Adam(model.classifier[-1].parameters(), 1e-5)
    criterion = nn.CrossEntropyLoss().cuda()

    # switch to train mode
    model.train()

    val_log = {
        'err': [],
        'ece': [],
        'mce': [],
        'loss': []
    }

    for epoch_idx in range(25):
        loss_val = 0.0
        for i, (input, target) in enumerate(train_loader):
            if torch.cuda.is_available():
                target = target.cuda()
                input_var = input.cuda()
                target_var = target
            else:
                target = target.cpu()
                input_var = input.cpu()
                target_var = target

            
            # compute output
            output_ = []
            for mc_run in range(mc_samples):
                output = model(input_var)
                output_.append(output)
            output = torch.mean(torch.stack(output_), dim=0)
            cross_entropy_loss = criterion(output, target_var)
            
            # Log loss
            loss = cross_entropy_loss

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val += loss.item()

        logits, labels = get_model_predictions(model, val_loader, 'cuda')
        val_err, val_ece, val_mce, val_loss = evaluate(softmax(logits), labels, verbose=False)  

        val_log['err'].append(val_err)
        val_log['ece'].append(val_ece)
        val_log['mce'].append(val_mce)
        val_log['loss'].append(val_loss)

        print("{:3d}    loss = {:.3f}    ece = {:.3f}    mce = {:.3f}    err = {:.3f}".format(epoch_idx,
            val_loss, val_ece, val_mce, val_err))

    return val_log


def main():

    device = 'cuda'

    # Data
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2471, 0.2435, 0.2616)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std)
    ])

    batch_size = 128
    trainset = CIFAR10(root="./../data/", train=True, transform=transform, download=False)
    trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=1)
    tr_eval_loader =  DataLoader(trainset, batch_size=2048, num_workers=1)
    valset = CIFAR10(root="./../data/", train=False, transform=transform, download=False)
    valloader = DataLoader(valset, batch_size=2048, num_workers=1)


    model = vgg.vgg11_bn(pretrained=True)
    model = model.to(device)

    # DNN model eval
    print("DNN model")
    logits, labels = get_model_predictions(model, tr_eval_loader, device)
    tr_err, tr_ece, tr_mce, tr_loss = evaluate(softmax(logits), labels, verbose=True)
    print("Validation:")
    logits, labels = get_model_predictions(model, valloader, device)
    val_err, val_ece, val_mce, val_loss = evaluate(softmax(logits), labels, verbose=True)
    print("--------------")

    # BNN using MOPED
    const_bnn_prior_parameters = {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0,
        "type": "Reparameterization",  # Flipout or Reparameterization
        "moped_enable": True,  # initialize mu/sigma from the dnn weights
        "moped_delta": 0.20,
    }
    dnn_to_bnn(model, const_bnn_prior_parameters)
    model = model.to(device)
    print("BNN Init model")
    logits, labels = get_bnn_model_predictions(model, tr_eval_loader, device)
    bnn_tr_err, bnn_tr_ece, bnn_tr_mce, bnn_tr_loss = evaluate(softmax(logits), labels, verbose=True)
    print("Validation:")
    logits, labels = get_model_predictions(model, valloader, device)
    bnn_val_err, bnn_val_ece, bnn_val_mce, bnn_val_loss = evaluate(softmax(logits), labels, verbose=True)
    print("--------------")

    # Train model
    print("Training....")
    train_log = train(model, trainloader, valloader)
    print("-------")

    print("BNN Trained model")
    logits, labels = get_bnn_model_predictions(model, tr_eval_loader, device)
    trained_tr_err, trained_tr_ece, trained_tr_mce, trained_tr_loss = evaluate(softmax(logits), labels, verbose=True)
    print("Validation:")
    logits, labels = get_model_predictions(model, valloader, device)
    trained_val_err, trained_val_ece, trained_val_mce, trained_val_loss = evaluate(softmax(logits), labels, verbose=True)
    print("--------------")

    # Save model
    torch.save(model.state_dict(), "cifar10-vgg11bn-moped-ll-1.pth")

    # Retrain without KL term
    retrain_log = retrain(model, trainloader, valloader)
    torch.save(model.state_dict(), "cifar10-vgg11bn-moped-ll-2.pth")

    # Save logs
    with open('retrain_logs.pkl', 'wb') as f:
        pickle.dump({
            'tr': {
                'err': tr_err,
                'ece': tr_ece,
                'mce': tr_mce,
                'loss': tr_loss
            },
            'val': {
                'err': val_err,
                'ece': val_ece,
                'mce': val_mce,
                'loss': val_loss
            },
            'bnn_tr': {
                'err': bnn_tr_err,
                'ece': bnn_tr_ece,
                'mce': bnn_tr_mce,
                'loss': bnn_tr_loss
            },
            'bnn_val': {
                'err': bnn_val_err,
                'ece': bnn_val_ece,
                'mce': bnn_val_mce,
                'loss': bnn_val_loss
            },
            'trained_tr': {
                'err': trained_tr_err,
                'ece': trained_tr_ece,
                'mce': trained_tr_mce,
                'loss': trained_tr_loss
            },
            'trained_val': {
                'err': trained_val_err,
                'ece': trained_val_ece,
                'mce': trained_val_mce,
                'loss': trained_val_loss
            },
            'train_log': train_log,
            'retrain_log': retrain_log
        }, f)


if __name__=="__main__":
    main()

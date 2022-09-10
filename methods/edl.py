"""
    Evidential Deep Learning for classfication
"""

import numpy as np
import torch
import torch.nn.functional as F

from pytorch_lightning import LightningModule
import torchmetrics


def onehotencoding(labels, num_classes):
    y = torch.eye(num_classes)
    y = y.type_as(labels)

    return y[labels]

def compute_prob_from_evidence(evidence_prior, evidence):
    alpha = evidence_prior + evidence # ypred is evidence
    prob = alpha / alpha.sum(dim=1).unsqueeze(-1)

    return prob

class EvidentialDeepLearning(LightningModule):
    def __init__(self, model, evidence_prior, class_weight=None, annealing_step=10) -> None:
        """

        """
        super().__init__()
        self.model = model
        if class_weight:
            self.register_buffer(name='class_weight', tensor=torch.tensor(class_weight))
        else:
            self.class_weight = None
        self.register_buffer(name='annealing_step', 
                                    tensor=torch.tensor(annealing_step))

        self.register_buffer(name='evidence_prior', tensor=torch.tensor(evidence_prior))
        #Inject evidence prior to model
        self.model.evidence_prior = self.evidence_prior
        
        # Metrics
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()

        if self.model.num_classes == 2:
            ignore_index = 0
        else:
            ignore_index = None
        self.train_f1score = torchmetrics.F1Score(ignore_index=ignore_index)
        self.val_f1score = torchmetrics.F1Score(ignore_index=ignore_index)
        self.test_f1score = torchmetrics.F1Score(ignore_index=ignore_index)

        # self.train_ece = torchmetrics.CalibrationError(n_bins=10, norm='l1')
        self.val_ece = torchmetrics.CalibrationError(n_bins=10, norm='l1')
        self.test_ece = torchmetrics.CalibrationError(n_bins=10, norm='l1')

        # Set loss function
        self.loss_function = self.edl_mse_loss


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        ypred = self(x)

        # Compute loss
        loss = self.loss_function(ypred, y)

        preds = torch.argmax(ypred, dim=1)

        self.log('train_loss', loss.detach())

        self.train_accuracy.update(preds, y)
        self.train_f1score.update(preds, y)

        return loss

    def on_train_epoch_end(self):
        self.log('train_acc', self.train_accuracy, prog_bar=True)
        self.log('train_f1', self.train_f1score, prog_bar=True)
        # self.log('train_ece', self.train_ece, prog_bar=True)

        return super().on_train_epoch_end()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        ypred = self(x)
        
        val_loss = self.loss_function(ypred, y)

        preds = torch.argmax(ypred, dim=1)

        # Compute prediction probabilities from evidence ypred
        pred_prob = self._compute_prob_from_evidence(ypred)
        
        self.log('val_loss', val_loss.detach())

        self.val_accuracy.update(preds, y)
        self.val_f1score.update(preds, y)
        self.val_ece.update(pred_prob, y)

        return {
            'loss': val_loss,
            'ypred': ypred
        }

    def validation_epoch_end(self, outputs):

        self.log('val_acc', self.val_accuracy, prog_bar=True)
        self.log('val_f1', self.val_f1score, prog_bar=True)
        self.log('val_ece', self.val_ece, prog_bar=True)

        # Log histogram of predictions
        ypred = torch.cat([o['ypred'] for o in outputs], dim=0)
        prob = self._compute_prob_from_evidence(ypred)
        if self.model.num_classes == 2:
            # For binary classification, only log score for class 1
            self.logger.experiment.add_histogram(
                'y_pred', prob[:, 1], # only for label 1
                global_step=self.current_epoch,
                bins=10)
        else:
            for i in range(self.model.num_classes):
                self.logger.experiment.add_histogram(
                    'y_pred_{:02d}'.format(i), 
                    prob[:, i], # for label `i`
                    global_step=self.current_epoch,
                    bins=10)

        return None

    def test_step(self, batch, batch_idx):
        x, y = batch
        
        ypred = self(x)
        
        test_loss = self.loss_function(ypred, y)
        preds = torch.argmax(ypred, dim=1)
        
        # Compute prediction probabilities from evidence ypred
        pred_prob = self._compute_prob_from_evidence(ypred)
        
        self.log('test_loss', test_loss.detach())

        # Compute metrics
        self.test_accuracy.update(preds, y)
        self.test_f1score.update(preds, y)
        self.test_ece.update(pred_prob, y)

        return {
            'loss': test_loss,
            'ypred': ypred
        }

    def test_epoch_end(self, outputs):

        test_acc = self.test_accuracy.compute()
        test_f1 = self.test_f1score.compute()
        test_ece = self.test_ece.compute()

        self.log('test_acc', test_acc, prog_bar=True)
        self.log('test_f1', test_f1, prog_bar=True)
        self.log('test_ece', test_ece, prog_bar=True)

        # Log histogram of predictions
        ypred = torch.cat([o['ypred'] for o in outputs], dim=0)
        prob = self._compute_prob_from_evidence(ypred)
        if self.model.num_classes == 2:
            # For binary classification, only log score for class 1
            self.logger.experiment.add_histogram(
                'y_pred_test', prob[:, 1], # only for label 1
                global_step=self.current_epoch,
                bins=10)
        else:
            for i in range(self.model.num_classes):
                self.logger.experiment.add_histogram(
                    'y_pred_test_{:02d}'.format(i), 
                    prob[:, i], # for label `i`
                    global_step=self.current_epoch,
                    bins=10)

        return None

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def _compute_prob_from_evidence(self, evidence):
        prob = compute_prob_from_evidence(self.evidence_prior, evidence)

        return prob

    def _loglikelihood_loss(self, target, alpha):
        S = torch.sum(alpha, dim=1, keepdim=True)

        ll_err = torch.sum((target - (alpha / S))**2, dim=1, keepdim=True)
        ll_var = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)

        ll = ll_err + ll_var

        return ll

    def _kl_divergence(self, alpha):
        ones = torch.ones([1, self.model.num_classes], dtype=torch.float32)
        ones = ones.type_as(alpha)
        sum_alpha = torch.sum(alpha, dim=1, keepdim=True)

        first_term = torch.lgamma(sum_alpha) \
                        - torch.lgamma(alpha).sum(dim=1, keepdim=True) \
                        + torch.lgamma(ones).sum(dim=1, keepdim=True) \
                        - torch.lgamma(ones.sum(dim=1, keepdim=True))

        second_term = (alpha - ones).mul(
                            torch.digamma(alpha) - torch.digamma(sum_alpha)
                        ).sum(dim=1, keepdim=True)

        kl = first_term + second_term

        return kl

    def _mse_loss(self, target, alpha):
        ll = self._loglikelihood_loss(target, alpha)

        annealing_coeff = torch.min(
            torch.tensor(1.0, dtype=torch.float32),
            torch.tensor(self.current_epoch / self.annealing_step, dtype=torch.float32)
        )

        kl_alpha = (alpha - 1) * (1 - target) + 1
        kl_div = annealing_coeff * self._kl_divergence(kl_alpha)

        return ll + kl_div

    def edl_mse_loss(self, evidence, target):
        """
            Compute MSE loss for EDL
        """
        alpha = evidence + self.evidence_prior
        target = onehotencoding(target, self.model.num_classes)
        loss = torch.mean(self._mse_loss(target, alpha))

        return loss

    @staticmethod
    def populate_missing_params(params, dataset):
        """
            
        """
        all_params = {}
        computed_params = {}
        K = dataset.n_labels
        

        eprior_arg = params.get('evidence_prior', 'uniform') # default is uniform prior
        if eprior_arg == 'uniform':
            computed_params['evidence_prior'] = np.ones(K) # uniform evidence prior
        elif eprior_arg == 'computed':
            # Compute frequency based prior - always less that 1
            if K == 2:
                p0 = dataset.n0 / (dataset.n0 + dataset.n1)
                computed_params['evidence_prior'] = np.array([p0, 1. - p0])
            else:
                try:
                    targets = dataset.ds.targets
                except AttributeError:
                    targets = [ dataset.ds.dataset.targets[i] for i in dataset.ds.indices ]
                except Exception as err:
                    raise err
                _, f = np.unique(targets, return_counts=True)
                f = f / f.sum()
                computed_params['evidence_prior'] = f
        elif isinstance(eprior_arg, int) or isinstance(eprior_arg, float):
            # Use this as the scaling
            computed_params['evidence_prior'] = eprior_arg * np.ones(K) # 
        else:
            raise ValueError(
                'Unknown evidence prior parameter `{}` for EDL'.format(eprior_arg)
            )
        
        # Remove evidence_prior param
        if 'evidence_prior' in params:
            del params['evidence_prior']

        all_params.update(computed_params)
        all_params.update(params)
        
        return all_params

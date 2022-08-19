import os
from turtle import forward

import torch
import torch.nn as nn
import torch.nn.functional as F

from bayesian_torch.layers import Conv2dReparameterization
from bayesian_torch.layers import LinearReparameterization
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss


class VGGBase(nn.Module):
    def __init__(self, features, K=10, init_weights=True):
        super().__init__()
        self.features = features
        # CIFAR 10 (7, 7) to (1, 1)
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            # nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, K),
        )
        self._initialize_weights()

    def forward(self, x):
        logits = self.get_logits(x)
        output = F.log_softmax(logits, dim=1)

        return output

    def get_logits(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)

        return logits

    def get_softmax(self, x):
        logits = self.get_logits(x)
        scores = F.softmax(logits, dim=1)

        return scores

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    "vgg3": [128, "M", 256, "M", 512, "M"], 
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512,
            512, 512, "M"],
    "vgg19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512,
            512, "M", 512, 512, 512, 512, "M"],
}


class VGG3(nn.Module):
    def __init__(self, K=10,
            prior_mu=0.0, prior_sigma=1.0,
            posterior_mu_init=0.0, posterior_rho_init=-3.0):
        super().__init__()
        self.model = VGGBase(make_layers(cfgs['vgg3']), K=K) # build deterministic model
        # script_dir = os.path.dirname(__file__)
        # state_dict = torch.load(script_dir + "/state_dicts/vgg11_bn.pt")
        # self.model.load_state_dict(state_dict)
        # convert to BNN
        dnn_to_bnn(self.model, {
            "prior_mu": prior_mu,
            "prior_sigma": prior_sigma,
            "posterior_mu_init": posterior_mu_init,
            "posterior_rho_init": posterior_rho_init,
            "type": "Reparameterization",
            "moped_enable": True,
            "moped_delta": 0.20
        })

        self.num_classes = K

    def forward(self, x, return_kl=True):
        out = self.model(x)

        if return_kl:
            kl = get_kl_loss(self.model)
            return out, kl

        return out


class VGG11(nn.Module):
    def __init__(self, K=10,
            prior_mu=0.0, prior_sigma=1.0,
            posterior_mu_init=0.0, posterior_rho_init=-3.0):
        super().__init__()
        self.model = VGGBase(make_layers(cfgs['vgg11']), K=K) # build deterministic model
        # script_dir = os.path.dirname(__file__)
        # state_dict = torch.load(script_dir + "/state_dicts/vgg11_bn.pt")
        # self.model.load_state_dict(state_dict)
        # convert to BNN
        dnn_to_bnn(self.model, {
            "prior_mu": prior_mu,
            "prior_sigma": prior_sigma,
            "posterior_mu_init": posterior_mu_init,
            "posterior_rho_init": posterior_rho_init,
            "type": "Reparameterization",
            "moped_enable": True,
            "moped_delta": 0.20
        })

        self.num_classes = K

    def forward(self, x, return_kl=True):
        out = self.model(x)

        if return_kl:
            kl = get_kl_loss(self.model)
            return out, kl

        return out

class VGG11Logits(VGG11):
    def __init__(self, K=10, prior_mu=0, prior_sigma=1, posterior_mu_init=0, posterior_rho_init=-3):
        super().__init__(K, prior_mu, prior_sigma, posterior_mu_init, posterior_rho_init)

    def forward(self, x, return_kl=True):
        out = self.model.get_logits(x)

        if return_kl:
            kl = get_kl_loss(self.model)
            return out, kl

        return out


class VGG11EDL(VGG11):
    def __init__(self, K=10):
        super().__init__()
        self.model = VGGBase(make_layers(cfgs['vgg11']), K=K) # build deterministic model
        self.num_classes = K

    def forward(self, x):
        out = self.get_evidence(x)

        return out

    def get_evidence(self, x):
        x = self.model.get_logits(x) # not really logits
        evidence = F.relu(x)

        return evidence

    def get_softmax(self, x):
        logits, _ = self.get_logits(x)
        scores = F.softmax(logits, dim=1)

        return scores


class VGG13(nn.Module):
    def __init__(self, K=10,
            prior_mu=0.0, prior_sigma=1.0,
            posterior_mu_init=0.0, posterior_rho_init=-3.0):
        super().__init__()
        self.model = VGGBase(make_layers(cfgs['vgg13']), K=K) # build deterministic model
        # convert to BNN
        dnn_to_bnn(self.model, {
            "prior_mu": prior_mu,
            "prior_sigma": prior_sigma,
            "posterior_mu_init": posterior_mu_init,
            "posterior_rho_init": posterior_rho_init,
            "type": "Reparameterization",
            "moped_enable": True,
            "moped_delta": 0.20
        })

        self.num_classes = K

    def forward(self, x, return_kl=True):
        out = self.model(x)

        if return_kl:
            kl = get_kl_loss(self.model)
            return out, kl

        return out


class VGG16(nn.Module):
    def __init__(self, K=10,
            prior_mu=0.0, prior_sigma=1.0,
            posterior_mu_init=0.0, posterior_rho_init=-3.0):
        super().__init__()
        self.model = VGGBase(make_layers(cfgs['vgg16']), K=K) # build deterministic model
        # convert to BNN
        dnn_to_bnn(self.model, {
            "prior_mu": prior_mu,
            "prior_sigma": prior_sigma,
            "posterior_mu_init": posterior_mu_init,
            "posterior_rho_init": posterior_rho_init,
            "type": "Reparameterization",
            "moped_enable": True,
            "moped_delta": 0.20
        })

        self.num_classes = K

    def forward(self, x, return_kl=True):
        out = self.model(x)

        if return_kl:
            kl = get_kl_loss(self.model)
            return out, kl

        return out


class VGG19(nn.Module):
    def __init__(self, K=10,
            prior_mu=0.0, prior_sigma=1.0,
            posterior_mu_init=0.0, posterior_rho_init=-3.0):
        super().__init__()
        self.model = VGGBase(make_layers(cfgs['vgg19']), K=K) # build deterministic model
        # convert to BNN
        dnn_to_bnn(self.model, {
            "prior_mu": prior_mu,
            "prior_sigma": prior_sigma,
            "posterior_mu_init": posterior_mu_init,
            "posterior_rho_init": posterior_rho_init,
            "type": "Reparameterization",
            "moped_enable": True,
            "moped_delta": 0.20
        })

        self.num_classes = K

    def forward(self, x, return_kl=True):
        out = self.model(x)

        if return_kl:
            kl = get_kl_loss(self.model)
            return out, kl

        return out

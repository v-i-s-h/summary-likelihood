import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss

from methods.edl import compute_prob_from_evidence


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

    def get_logits(self, x):
        return self.model.get_logits(x)

    def get_softmax(self, x):
        return self.model.get_softmax(x)


class VGG11EDL(VGG11):
    def __init__(self, K=10):
        super().__init__()
        self.model = VGGBase(make_layers(cfgs['vgg11']), K=K) # build deterministic model
        self.num_classes = K

        # Note: evidence prior is defined here, but will be injected into
        # the model by edl method
        self.evidence_prior = None

    def forward(self, x):
        out = self.get_evidence(x)

        return out

    def get_evidence(self, x):
        x = self.model.get_logits(x) # not really logits in case of EDL, but pre-evidence
        evidence = F.relu(x)

        return evidence

    def get_softmax(self, x):
        evidence = self.get_evidence(x)
        if self.evidence_prior is not None:
            scores = compute_prob_from_evidence(self.evidence_prior, evidence)
        else:
            print('WARNING: Unknown evidence prior for EDL model. Using uniform evidence')
            self.evidence_prior = torch.ones(self.num_classes, device=x.device)
            scores = compute_prob_from_evidence(self.evidence_prior, evidence)

        return scores


class VGG11Deterministic(nn.Module):
    def __init__(self, K=10, pretrained=False, head_only=False):
        super().__init__()
        self.model = VGGBase(make_layers(cfgs['vgg11']), K=K) # build deterministic model

        if pretrained:
            script_dir = os.path.dirname(__file__)
            state_dict = torch.load(script_dir + "/state_dicts/vgg11_bn.pt")
            self.model.load_state_dict(state_dict)
            if head_only:
                # Reset Linear layers
                for m in self.model.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, 0, 0.01)
                        nn.init.constant_(m.bias, 0)
                print("===> Reseting linear layers")
            print("===> Pretrained model", "(head only)" if head_only  else "")
        
        self.num_classes = K

    def forward(self, x):
        out = self.model(x) # log_softmax

        return out

    def get_logits(self, x):
        return self.model.get_logits(x)

    def get_softmax(self, x):
        return self.model.get_softmax(x)

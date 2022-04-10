import torch.nn.functional as F

from bayesian_torch.models.bayesian.resnet_variational import BasicBlock, ResNet


class ResNet20(ResNet):
    def __init__(self, K=10):
        super().__init__(BasicBlock, [3, 3, 3], num_classes=K)

    def forward(self, x):
        x, kl_sum = super().forward(x)

        output = F.log_softmax(x, dim=1)

        return output, kl_sum


if __name__ == "__main__":
    import torch
    
    m = ResNet20(K=10)
    x = torch.rand((1, 3, 32, 32))
    y, kl = m(x)
    print(y, kl)
    y = torch.exp(y)
    print(y)
    print(y.sum())

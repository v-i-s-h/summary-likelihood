"""
    Dataset transforms
"""

from torchvision import transforms


def normalize_x(mean=[0.0], std=[1.0]):
    """
        Normalize inputs to zero mean unit deviation
    
    Parameters
    ----------
    mean : list
        List of mean value
    std : list
        List of std deviation value

    Returns
    -------
    x_transform - input transforms
    """

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

def normalize_x_cifar():
    # return normalize_x(
    #     mean=(0.4914, 0.4822, 0.4465),
    #     std=(0.2471, 0.2435, 0.2616)
    # )
    return normalize_x()

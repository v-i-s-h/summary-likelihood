"""
    Dataset transforms
"""

from torchvision import transforms


def normalize_x(mean=[0.0], std=[1.0]):
    """
        Normalize inputs
    
    Parameters
    ----------
    mean : list
        List of mean value in each dimension
    std : list
        List of std deviation value in each dimension

    Returns
    -------
    x_transform - input transforms
    """

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def normalize_x_cifar():
    return normalize_x()


def normalize_x_cifar_v2():
    return normalize_x(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2471, 0.2435, 0.2616)
    )

def cifar_da_x():
    return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2471, 0.2435, 0.2616)    
                ),
            ])


def tensorize():
    return transforms.Compose([
        transforms.ToTensor()
    ])


def normalize_x_sst():
    # Placeholder for normalization
    # Embeddings are used - so no normalization to be applied
    return None

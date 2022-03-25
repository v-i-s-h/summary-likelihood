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
        transforms.Normalize(mean=[0.0], std=[1.0])
    ])

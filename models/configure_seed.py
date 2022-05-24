import torch, numpy, random


def configure_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    numpy.random.seed(seed)
    return None
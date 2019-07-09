from sres.models import *
from sres.datasets import *
from sres.solvers import *
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR
from torch.nn import MSELoss


MODELS = {
    'srresnet': SRResNet,
    'srgan': [SRGenerator, SRDiscriminator],
    'esrgan': [ESRGenerator, ESRDiscriminator],
    }

DATASETS = {
    'div2k': Div2K,
    'df2k': DF2K
}

OPTS = {
    'adam': Adam
}

LOSSES = {
    'mse': MSELoss
}

SCHEDULERS = {
    'reduce': ReduceLROnPlateau,
    'cyclic': CyclicLR
}

SOLVERS = {
    'std': StandardSolver,
    'gan': GANSolver
}
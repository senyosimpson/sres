from sres.models import *
from sres.datasets import *
from sres.solvers import *
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CyclicLR
from torch.nn import MSELoss, BCELoss
from sres.losses import PerceptualLoss, VGGLoss


MODELS = {
    'srresnet': SRResNet,
    'msrresnet': MSRResNet,
    'srgan': [SRGenerator, SRDiscriminator],
    'esrgan': [ESRGenerator, ESRDiscriminator],
    }

DATASETS = {
    'div2k': Div2K,
    'df2k': DF2K,
    'set5': Set5,
    'set14': Set14,
    'bsd100': BSD100,
    'urban100': Urban100,
    'pirm_val': PIRMVal,
    'div2k_val': Div2KVal
}

OPTS = {
    'adam': Adam
}

LOSSES = {
    'mse': MSELoss,
    'bce': BCELoss,
    'perceptual': PerceptualLoss,
    'vgg': VGGLoss
}

SCHEDULERS = {
    'reduce': ReduceLROnPlateau,
    'cyclic': CyclicLR
}

SOLVERS = {
    'std': StandardSolver,
    'gan': GANSolver
}

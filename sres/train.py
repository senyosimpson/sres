import shutil
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from sres.transforms import ToTensor, Normalize, Transpose, \
    RandomCrop, RandomHorizontalFlip, RandomRotateNinety
from sres.utils import Config
from sres.constants import MODELS, DATASETS, OPTS, LOSSES, SCHEDULERS, SOLVERS
from torch.nn import MSELoss
from datetime import datetime


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        type=str,
                        required=True,
                        help='path to the config file')
    parser.add_argument('--root-dir',
                        type=str,
                        required=True,
                        help='root directory for dataset')
    parser.add_argument('--logdir',
                        type=str,
                        required=True,
                        help='path to save checkpoints')
    args = parser.parse_args()

    use_cuda = not False and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # copy config to log dir
    shutil.copy(args.config, args.logdir)
    # unpack config
    conf = Config(path=args.config)
    model_name = conf['model']
    solver_type = conf['solver']
    epochs = conf['epochs']
    loss_fn = conf['loss_fn']
    ds_name = conf['dataset_name']
    ds_params = conf['dataset_params']
    opt_name = conf['optimizer_name']
    opt_params = conf['optimizer_params']
    sched_name = conf['scheduler_name']
    sched_params = conf['scheduler_params']
    patch_size = conf['patch_size']
    checkpoint = conf['checkpoint']

    model = MODELS[model_name]
    model = model().to(device)

    opt = OPTS[opt_name]
    optimizer = opt(model.parameters(), **opt_params)

    use_sched = False
    scheduler = None
    if sched_name and sched_params:
        sched = SCHEDULERS[sched_name]
        scheduler = sched(optimizer, **sched_params)
        use_sched = True
        
    loss_fn = LOSSES[loss_fn]
    loss_fn = loss_fn()

    if checkpoint:
        chkpt = torch.load(checkpoint)
        model.load_state_dict(chkpt['model_state_dict'])
        optimizer.load_state_dict(chkpt['optimizer_state_dict'])
        start_epoch = chkpt['epoch']
        loss = chkpt['loss']

    tsfm = Compose([
        RandomCrop(patch_size),
        RandomHorizontalFlip(),
        RandomRotateNinety(),
        ToTensor()
    ])

    ds = DATASETS[ds_name]
    dataset = ds(args.root_dir, transform=tsfm)
    dataloader =  DataLoader(dataset, **ds_params)

    solver = SOLVERS[solver_type]
    solver = solver(model, optimizer, loss_fn, dataloader, scheduler, checkpoint)
    batch_size = ds_params['batch_size']
    solver.solve(epochs, batch_size, args.logdir)

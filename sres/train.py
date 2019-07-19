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
    generator_path = conf['generator_path']

    model = MODELS[model_name]
    if isinstance(model, list):
        generator, discriminator = MODELS[model_name]
        generator = generator().to(device)
        discriminator = discriminator(input_shape=patch_size).to(device)
    else:
        model = model().to(device)

    opt = OPTS[opt_name]
    if 'gan' in model_name:
        gen_optimizer = opt(generator.parameters(), **opt_params)
        discrim_optimizer = opt(discriminator.parameters(), **opt_params)
        optimizer = [gen_optimizer, discrim_optimizer]
    else:
        optimizer = opt(model.parameters(), **opt_params)


    scheduler = None
    if sched_name and sched_params:
        sched = SCHEDULERS[sched_name]
        scheduler = sched(optimizer, **sched_params)
    
    if isinstance(loss_fn, list):
        gen_loss_name, discrim_loss_name = loss_fn
        gen_loss = LOSSES[gen_loss_name]
        discrim_loss = LOSSES[discrim_loss_name]

        gen_loss = gen_loss()
        discrim_loss = discrim_loss()
        loss_fn = [gen_loss, discrim_loss]
    else:
        loss_fn = LOSSES[loss_fn]
        loss_fn = loss_fn()

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
    if 'gan' in model_name:
        solver = solver(conf, generator, discriminator, optimizer, loss_fn, dataloader, generator_path, scheduler)
    else:
        solver = solver(conf, model, optimizer, loss_fn, dataloader, scheduler)
    batch_size = ds_params['batch_size']
    solver.solve(epochs, batch_size, args.logdir, checkpoint)

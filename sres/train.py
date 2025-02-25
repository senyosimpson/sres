import shutil
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from sres.utils import Config
from sres.transforms import ToTensor
from sres.transforms import RandomCrop
from sres.transforms import RandomHorizontalFlip
from sres.transforms import RandomRotateNinety
from sres.constants import MODELS
from sres.constants import DATASETS
from sres.constants import OPTS
from sres.constants import LOSSES
from sres.constants import SCHEDULERS
from sres.constants import SOLVERS


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        type=str,
                        required=True,
                        help='path to the config file')
    args = parser.parse_args()

    use_cuda = not False and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # unpack config
    conf = Config(path=args.config)
    logdir = conf['logdir']
    model_name = conf['model']
    solver_type = conf['solver']
    epochs = conf['epochs']
    loss_fn = conf['loss_fn']
    ds_root = conf['dataset_root']
    ds_name = conf['dataset_name']
    ds_params = conf['dataset_params']
    opt_name = conf['optimizer_name']
    opt_params = conf['optimizer_params']
    sched_name = conf['scheduler_name']
    sched_params = conf['scheduler_params']
    patch_size = conf['patch_size']
    checkpoint = conf['checkpoint']
    generator_path = conf['generator_path']

    # copy config to log dir
    shutil.copy(args.config, logdir)

    model = MODELS[model_name]
    opt = OPTS[opt_name]
    if solver_type == 'gan':
        # model
        generator, discriminator = model
        generator = generator().to(device)
        discriminator = discriminator(input_shape=patch_size).to(device)
        # optimizer
        gen_optimizer = opt(generator.parameters(), **opt_params)
        discrim_optimizer = opt(discriminator.parameters(), **opt_params)
        optimizer = [gen_optimizer, discrim_optimizer]
        # loss function
        gen_loss_name, discrim_loss_name = loss_fn
        gen_loss = LOSSES[gen_loss_name]
        gen_loss = gen_loss()
        discrim_loss = LOSSES[discrim_loss_name]
        discrim_loss = discrim_loss()
        loss_fn = [gen_loss, discrim_loss]
    else:
        # model
        model = model().to(device)
        # optimizer
        optimizer = opt(model.parameters(), **opt_params)
        # loss function
        loss_fn = LOSSES[loss_fn]
        loss_fn = loss_fn()


    scheduler = None
    if sched_name and sched_params:
        sched = SCHEDULERS[sched_name]
        scheduler = sched(optimizer, **sched_params)

    tsfm = Compose([
        RandomCrop(patch_size),
        RandomHorizontalFlip(),
        RandomRotateNinety(),
        ToTensor()
    ])
    ds = DATASETS[ds_name]
    dataset = ds(ds_root, transform=tsfm)
    dataloader = DataLoader(dataset, **ds_params)

    solver = SOLVERS[solver_type]
    if solver_type == 'gan':
        solver = solver(conf, generator, discriminator, optimizer, loss_fn, dataloader, generator_path, scheduler)
    else:
        solver = solver(conf, model, optimizer, loss_fn, dataloader, scheduler)

    batch_size = ds_params['batch_size']
    solver.solve(epochs, batch_size, logdir, checkpoint)

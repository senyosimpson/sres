import os
import argparse
import logging
import torch
import torch.optim as optim
from torch.utils.Data import DataLoader
from torchvision.transforms import Compose
from sres.transforms import ToTensor, Normalize, Transpose
from sres.datasets import PIRM
from sres.models import srgan
from sres.losses import PerceptualLoss
from torch.nn import BCELoss
from datetime import datetime


if __name__ == '__main__':
    date = datetime.today().strftime('%m_%d')
    logger = logging.getLogger('dir')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir',
                        type=str,
                        required=True,
                        help='root directory for dataset')
    parser.add_argument('--load-checkpoint',
                        type=str,
                        required=False,
                        help='path to checkpoint to load')
    parser.add_argument('--epochs',
                        type=int,
                        required=False,
                        default=10,
                        help='the number of epochs')
    parser.add_argument('--batch-size',
                        type=int,
                        required=False,
                        default=32,
                        help='the batch size')
    parser.add_argument('--logdir',
                        type=str,
                        required=True,
                        help='path to save checkpoints')
    parser.add_argument('--num-workers',
                        type=int,
                        required=False,
                        default=4,
                        help='number of workers for data loading')
    args = parser.parse_args()

    logger.info('MAIN SCRIPT STARTED')
    use_cuda = not False and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    generator = srgan.Generator()
    discriminator = srgan.Discriminator(input_shape=(96, 96))
    generator.to(device); discriminator.to(device)

    optimizer_gen = optim.Adam(generator.parameters())
    optimizer_disc = optim.Adam(discriminator.parameters())

    if args.load_checkpoint:
        chkpt = torch.load(args.load_checkpoint)
        generator.load_state_dict(chkpt['generator_state_dict'])
        discriminator.load_state_dict(chkpt['discriminator_state_dict'])
        optimizer_gen.load_state_dict(chkpt['optimizer_gen_state_dict'])
        optimizer_disc.load_state_dict(chkpt['optimizer_disc_state_dict'])
        start_epoch = chkpt['epoch']
        loss = chkpt['loss']

    # losses
    gen_loss = PerceptualLoss()
    discrim_loss = BCELoss()

    tsfm = Compose([
        Transpose(),
        Normalize(),
        ToTensor()
    ])

    pirm = PIRM(args.root_dir, transform=tsfm)
    dataloader =  DataLoader(
        pirm,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # details of training
    logger.info('')
    logger.info('Batch Size : %d' % args.batch_size)
    logger.info('Number of Epochs : %d' % args.epochs)
    logger.info('Steps per Epoch : %d' % len(dataloader))
    logger.info('')

    generator.train()
    discriminator.train()
    start_epoch = start_epoch if start_epoch else 0

    for epoch in range(start_epoch, args.epochs):
        logger.info('============== Epoch %d/%d ==============' % (epoch+1, args.epochs))
        for step, image_pair in enumerate(dataloader):
            lres_img, hres_img = image_pair
            lres_img.to(device); hres_img.to(device)

            generated_img = generator(lres_img)
            prediction_generated = discriminator(generated_img)
            prediction_real = discriminator(hres_img)

            # train discriminator
            optimizer_disc.zero_grad()
            target_gen = 0.7 + (torch.rand(args.batch_size, 1) * 0.5) # between 0.7 - 1.2
            target_real = torch.rand(args.batch_size, 1) * 0.3 # between 0.0 - 0.3

            target_gen.to(device); target_real.to(device)
            d_loss = discrim_loss(prediction_generated, target_gen) + discrim_loss(prediction_real, target_real)
            d_loss.backwards()
            optimizer_disc.step()

            # train generator
            optimizer_gen.zero_grad()
            g_loss = gen_loss(generated_img, hres_img, prediction_generated)
            g_loss.backwards()
            optimizer_gen.step()

            logger.info('step: %d, Gen loss: %.3f, Discrim Loss: %.3f' % (step, g_loss.item(), d_loss.item()))


        save_path = '%s_checkpoint_%d_%s%s' % (srgan.MODEL_NAME, epoch+1, date, '.pt')
        torch.save({
            'epoch': epoch,
            'loss': loss,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_gen_state_dict': optimizer_gen.state_dict(),
            'optimizer_disc_state_dict': optimizer_disc.state_dict()},
            f = os.path.join(args.logdir, save_path))
        logger.info('Checkpoint saved to %s' % save_path)

    logger.info('Training Complete')
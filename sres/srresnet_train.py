import os
import argparse
import logging
import torch
import torch.optim as optim
from torch.utils.Data import DataLoader
from torchvision.transforms import Compose
from sres.transforms import ToTensor, Normalize, Transpose, \
    RandomCrop, RandomHorizontalFlip, RandomRotateNinety
from sres.datasets import Div2K
from sres.models.srresnet import SRResNet
from torch.nn import MSELoss
from datetime import datetime


if __name__ == '__main__':
    date = datetime.today().strftime('%m_%d')
    logger = logging.getLogger('srresnet')
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
                        default=16,
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

    model = SRResNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))
    mse = MSELoss()

    if args.load_checkpoint:
        chkpt = torch.load(args.load_checkpoint)
        model.load_state_dict(chkpt['model_state_dict'])
        optimizer.load_state_dict(chkpt['optimizer_state_dict'])
        start_epoch = chkpt['epoch']
        loss = chkpt['loss']

    tsfm = Compose([
        Transpose(),
        Normalize(),
        RandomCrop(96),
        RandomHorizontalFlip(),
        RandomRotateNinety(),
        ToTensor()
    ])
    div2k = Div2K(args.root_dir, transform=tsfm)
    dataloader =  DataLoader(
        div2k,
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

    model.train()
    start_epoch = start_epoch if start_epoch else 0
    for epoch in range(start_epoch, args.epochs):
        logger.info('============== Epoch %d/%d ==============' % (epoch+1, args.epochs))
        mean_loss = 0
        for step, image_pair in enumerate(dataloader):
            lres_img, hres_img = image_pair
            lres_img.to(device); hres_img.to(device)

            generated_img = model(lres_img)
            optimizer.zero_grad()
            loss = mse(generated_img, hres_img)
            loss.backward()
            optimizer.step()

            logger.info('step: %d, loss: %.3f' % (step, loss.item()))
            mean_loss += loss.item()

        logger.info('epoch : %d, average loss : %.3f' % (epoch+1, mean_loss/len(dataloader)))

        save_path = '%s_chkpt_%d_%s%s' % (model.name, epoch+1, date, '.pt')
        torch.save({
            'epoch': epoch,
            'loss': loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()},
            f = os.path.join(args.logdir, save_path))
        logger.info('Checkpoint saved to %s' % save_path)

    logger.info('Training Complete')
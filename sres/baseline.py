import logging
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from sres.transforms import ToTensor
from sres.constants import DATASETS
from sres.metrics import SSIM
from sres.metrics import PSNR

if __name__ == '__main__':
    logger = logging.getLogger('dir')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        type=str,
                        required=True,
                        help='the dataset to evaluate model on')
    parser.add_argument('--root-dir',
                        type=str,
                        required=True,
                        help='the path to the root directory of the dataset')
    parser.add_argument('--batch-size',
                        type=int,
                        default=1,
                        required=False,
                        help='the batch size')
    parser.add_argument('--num-workers',
                        type=int,
                        default=4,
                        required=False,
                        help='number of workers used in loading data')
    args = parser.parse_args()
    root_dir = args.root_dir
    dataset = args.dataset
    batch_size = args.batch_size
    num_workers = args.num_workers
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tsfm = Compose([
        ToTensor()
    ])

    ds = DATASETS[dataset]
    dataset = ds(root_dir, transform=tsfm)
    dataloader = DataLoader(dataset,
                            shuffle=False,
                            batch_size=batch_size,
                            num_workers=num_workers)

    psnr = PSNR()
    ssim = SSIM()

    logger.info('============== Starting Evaluation ==============')
    with torch.no_grad():
        avg_psnr_nn = 0
        avg_psnr_bicubic = 0
        avg_ssim_nn = 0
        avg_ssim_bicubic = 0
        for idx, image_pair in enumerate(dataloader):
            logger.info('Batch %d/%d' % (idx + 1, len(dataloader)))
            lr_image, hr_image = image_pair
            lr_image = lr_image.to(device)
            hr_image = hr_image.to(device)
            generated_img = F.interpolate(lr_image, scale_factor=4, mode='nearest')
            generated_img = torch.clamp(generated_img, min=0., max=1.)
            # crop border pixels
            hr_image = hr_image[:, :, 4:-4, 4:-4]
            generated_img = generated_img[:, :, 4:-4, 4:-4]
            # evaluate metrics
            avg_psnr_nn += psnr(generated_img, hr_image)
            avg_ssim_nn += ssim(generated_img, hr_image)

            generated_img = F.interpolate(lr_image, scale_factor=4, mode='bicubic')
            generated_img = torch.clamp(generated_img, min=0., max=1.)
            # crop border pixels
            generated_img = generated_img[:, :, 4:-4, 4:-4]
            # evaluate metrics
            avg_psnr_bicubic += psnr(generated_img, hr_image)
            avg_ssim_bicubic += ssim(generated_img, hr_image)

    print("Avg. PSNR Nearest: {:.4f} dB".format(avg_psnr_nn / len(dataloader)))
    print("Avg. SSIM Nearest: {:.4f} dB".format(avg_ssim_nn / len(dataloader)))
    print("Avg. PSNR Bicubic: {:.4f} dB".format(avg_psnr_bicubic / len(dataloader)))
    print("Avg. SSIM Bicubic: {:.4f} dB".format(avg_ssim_bicubic / len(dataloader)))

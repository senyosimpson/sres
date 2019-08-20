import logging
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from sres.transforms import ToTensor, Normalize, Transpose, \
    RandomCrop, RandomHorizontalFlip, RandomRotateNinety
from sres.constants import MODELS, DATASETS
from sres.metrics import SSIM, PSNR


if __name__ == '__main__':
    logger = logging.getLogger('dir')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        type=str,
                        required=True,
                        help='the model to evaluate')
    parser.add_argument('--model-path',
                        type=str,
                        required=True,
                        help='path to model weights')
    parser.add_argument('--dataset',
                        type=str,
                        required=True,
                        help='the dataset to evaluate model on')
    parser.add_argument('--root-dir',
                        type=str,
                        required=True,
                        help='the path to the root directory of the dataset')
    parser.add_argument('--colour-space',
                        type=str,
                        default='rgb',
                        required=False,
                        help='the colour space to evaluate on')
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

    use_cuda = not False and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    weights = torch.load(args.model_path)['model_state_dict']
    model = MODELS[args.model]
    model = model()
    model.load_state_dict(weights)
    model.to(device)
    
    tsfm = Compose([
        ToTensor()
    ])

    ds = DATASETS[args.dataset]
    dataset = ds(args.root_dir, transform=tsfm)
    dataloader = DataLoader(dataset,
                            shuffle=False,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers)

    psnr = PSNR()
    ssim = SSIM()

    logger.info('============== Starting Evaluation ==============')
    model.eval()
    with torch.no_grad():
        avg_psnr = 0
        for idx, image_pair in enumerate(dataloader):
            logger.info('Batch %d/%d' % (idx+1, len(dataloader)))
            lr_image, hr_image = image_pair
            generated_img = model(lr_image)
            avg_psnr += psnr(generated_img, hr_image)
    
    print("Avg. PSNR: {:.4f} dB".format(avg_psnr / len(dataloader)))
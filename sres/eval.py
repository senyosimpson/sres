import logging
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from sres.transforms import ToTensor
from sres.constants import MODELS
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
    model = args.model
    model_path = args.model_path
    root_dir = args.root_dir
    dataset = args.dataset
    batch_size = args.batch_size
    num_workers = args.num_workers
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # make a utility function called get_attr
    if model == 'srgan':
        weights = torch.load(model_path)['generator_state_dict']
        model = MODELS[model][0]
    else:
        weights = torch.load(model_path)['model_state_dict']
        model = MODELS[model]

    model = model().to(device)
    model.load_state_dict(weights)

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
    model.eval()
    with torch.no_grad():
        avg_psnr = 0
        avg_ssim = 0
        for idx, image_pair in enumerate(dataloader):
            logger.info('Batch %d/%d' % (idx+1, len(dataloader)))
            lr_image, hr_image = image_pair
            lr_image = lr_image.to(device)
            hr_image = hr_image.to(device)
            generated_img = model(lr_image)
            # crop border pixels
            hr_image = hr_image[:, :, 4:-4, 4:-4]
            generated_img = generated_img[:, :, 4:-4, 4:-4]
            # evaluate metrics
            avg_psnr += psnr(generated_img, hr_image)
            avg_ssim += ssim(generated_img, hr_image)
    
    print("Avg. PSNR: {:.4f} dB".format(avg_psnr / len(dataloader)))
    print("Avg. SSIM: {:.4f} dB".format(avg_ssim / len(dataloader)))

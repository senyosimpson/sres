import os
from PIL import Image
from torch.utils.data import Dataset
from glob import glob

TRAIN_DIR_NAMES = ['DIV2K_train_LR_bicubic_X4', 'DIV2K_train_HR'] 
VALID_DIR_NAMES = ['DIV2K_valid_LR_bicubic_X4', 'DIV2K_valid_HR', ]


class Div2K(Dataset):
    def __init__(self, root, training=True, fmat='png', transform=None):
        ext = '*.%s' % fmat
        lr_dir, hr_dir = TRAIN_DIR_NAMES
        if not training:
            lr_dir, hr_dir = VALID_DIR_NAMES
        self.lres_images = sorted(glob(os.path.join(root, lr_dir, ext)))
        self.hres_images = sorted(glob(os.path.join(root, hr_dir, ext)))
        self.dataset = list(zip(self.lres_images, self.hres_images))
        self.transform = transform

    def __getitem__(self, idx):
        lres_img_path, hres_img_path = self.dataset[idx]
        lres_img = Image.open(lres_img_path)
        hres_img = Image.open(hres_img_path)
        sample = (lres_img, hres_img)

        if self.transform:
            sample = self.transform(sample)
        return sample
        
    def __len__(self):
        return len(self.dataset)
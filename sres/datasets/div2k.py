import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from glob import glob

DIR_NAMES = ['DIV2K_train_LR_bicubic_X4', 'DIV2K_train_HR']


class Div2K(Dataset):
    def __init__(self, root, fmat='png', transform=None):
        ext = '*.%s' % fmat
        self.lres_images = sorted(glob(os.path.join(root, DIR_NAMES[0], ext)))
        self.hres_images = sorted(glob(os.path.join(root, DIR_NAMES[1], ext)))
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
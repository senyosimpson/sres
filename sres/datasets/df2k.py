import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from glob import glob

DIR_NAMES = {'lr_dirname':'LR', 'hr_dirname':'HR'}

class DF2K(Dataset):
    def __init__(self, root, fmat='png', transform=None):
        self.root = root
        self.ext = '*.%s' % fmat
        self.hres_images = glob(os.path.join(self.root, DIR_NAMES['hr_dirname'], self.ext))
        self.lres_images = self._get_lr_paths(self.hres_images)
        self.dataset = list(zip(self.lres_images, self.hres_images))
        self.transform = transform
        
    def _get_lr_paths(self, hr_paths):
        lr_paths = []
        for path in hr_paths:
            root, ext = os.path.splitext(path)
            basename = os.path.basename(root)
            lr_filename = '%sx4%s' % (basename, ext)
            lr_path = os.path.join(self.root, DIR_NAMES['lr_dirname'], lr_filename)
            lr_paths.append(lr_path)
        return lr_paths

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
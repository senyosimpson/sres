import os
import skimage
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from glob import glob

DIR_NAMES = ['4x_downsampled', 'Original']

class PIRM(Dataset):
    def __init__(self, root, fmat='png', transform=None):
        """
        args:
            root (str) : path to the root directory containing images
            fmat (str) : the format of the images in the dataset
        """
        ext = '*.%s' % fmat
        self.lres_images = sorted(glob(os.path.join(root, DIR_NAMES[0], ext)))
        self.hres_images = sorted(glob(os.path.join(root, DIR_NAMES[1], ext)))
        self.dataset = list(zip(self.lres_images, self.hres_images))
        self.transform = transform
    
    def __getitem__(self, idx):
        lres_img_path, hres_img_path = self.dataset[idx]
        lres_img = skimage.io.imread(lres_img_path)
        hres_img = skimage.io.imread(hres_img_path)
        sample = (lres_img, hres_img)

        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        len(self.dataset)
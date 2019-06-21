import os
import skimage
import random
import torch
from torch.utils.data import Dataset
from glob import glob

class PIRM(Dataset):
    def __init__(self, root, fmat='png', transform=None):
        """
        args:
            root (str) : path to the root directory containing images
            fmat (str) : the format of the images in the dataset
        """
        ext = '*.%s' % fmat
        self.image_paths = glob(os.path.join(root, ext))
        self.transform = transform
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        sample = skimage.io.imread(image_path)
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        len(self.image_paths)
""" Transforms on images used in the super resolution processing pipeline """
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
import random


class ToTensor:
    def __call__(self, sample):
        lres_img, hres_img = sample
        sample = (torch.Tensor(lres_img), torch.Tensor(hres_img))
        return sample


class Transpose:
    def __call__(self, sample):
        lres_img, hres_img = sample
        lres_img = lres_img.transpose((2,0,1))
        hres_img = hres_img.transpose((2,0,1))
        sample = (lres_img, hres_img)
        return sample

    
class Normalize:
    def __call__(self, sample):
        lres_img, hres_img = sample
        lres_img = (lres_img - lres_img.min()) / (lres_img.max() - lres_img.min())
        hres_img = 2 * (hres_img - hres_img.min()) / (hres_img.max() - hres_img.min())-1
        sample = (lres_img, hres_img)
        return sample


class RandomCrop:
    def __init__(self, output_size, scale=4):
        self.output_size = output_size
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        self.scale = scale

    def __call__(self, sample):
        lres_img, hres_img = sample
        lr_dim = self.output_size[0] // self.scale
        lres_crop_size = (lr_dim, lr_dim)
        lres_params = transforms.RandomCrop.get_params(
            lres_img,
            output_size=lres_crop_size
        )
        hres_params = [param * self.scale for param in lres_params]
        lres_img_cropped = TF.crop(lres_img, *lres_params)
        hres_img_cropped = TF.crop(hres_img, *hres_params)
        sample = (lres_img_cropped, hres_img_cropped)
        return sample


class RandomRotateNinety:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        lres_img, hres_img = sample
        if random.random() > self.p:
            degrees = random.choice([90, 270])
            lres_img_rotated = TF.rotate(lres_img, degrees, resample='PIL.Image.BICUBIC')
            hres_img_rotated = TF.rotate(hres_img, degrees, resample='PIL.Image.BICUBIC')
        
        sample = (lres_img_rotated, hres_img_rotated)
        return sample


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        lres_img, hres_img = sample
        if random.random() > self.p:
            lres_img_flipped = TF.hflip(lres_img)
            hres_img_flipped = TF.hflip(hres_img)
        
        sample = (lres_img_flipped, hres_img_flipped)
        return sample

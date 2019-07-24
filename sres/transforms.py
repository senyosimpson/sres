""" Transforms on images used in the super resolution processing pipeline """
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import random


class ToTensor:
    def __call__(self, sample):
        lres_img, hres_img = sample
        sample = (TF.to_tensor(lres_img), TF.to_tensor(hres_img))
        return sample


class Transpose:
    def __call__(self, sample):
        lres_img, hres_img = sample
        lres_img = lres_img.transpose((2,0,1))
        hres_img = hres_img.transpose((2,0,1))
        sample = (lres_img, hres_img)
        return sample

class Standardize:
    def __call__(self, sample):
        lres_img, hres_img = sample
        lres_img = (lres_img - lres_img.min()) / (lres_img.max() - lres_img.min())
        hres_img = (hres_img - hres_img.min()) / (hres_img.max() - hres_img.min())
        sample = (lres_img, hres_img)
        return sample


class Denormalize:
    def __call__(self, sample, mean, std):
        lres_img, hres_img = sample
        mean = torch.as_tensor(mean, dtype=torch.float32, device=lres_img.device)
        std = torch.as_tensor(std, dtype=torch.float32, device=lres_img.device)
        lres_img.mul_(std[:, None, None]).add_(mean[:, None, None])
        hres_img.mul_(std[:, None, None]).add_(mean[:, None, None])
        sample = (lres_img, hres_img)
        return sample


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        lres_img, hres_img = sample
        lres_img = TF.normalize(lres_img, self.mean, self.std)
        hres_img = TF.normalize(hres_img, self.mean, self.std)
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
            lres_img = TF.rotate(lres_img, degrees, resample=Image.BICUBIC)
            hres_img = TF.rotate(hres_img, degrees, resample=Image.BICUBIC)
        
        sample = (lres_img, hres_img)
        return sample


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        lres_img, hres_img = sample
        if random.random() > self.p:
            lres_img = TF.hflip(lres_img)
            hres_img = TF.hflip(hres_img)
        
        sample = (lres_img, hres_img)
        return sample

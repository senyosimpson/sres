import os
from PIL import Image
from torch.utils.data import Dataset
from glob import glob


class Set5(Dataset):
    def __init__(self, root, fmat='png', transform=None):
        ext = '*.%s' % fmat
        self.dataset = self.build_dataset(os.path.join(root, ext))
        self.transform = transform

    def build_dataset(self, root):
        image_paths = glob(root)
        lr_images = sorted([path for path in image_paths if 'LR' in path])
        hr_images = sorted([path for path in image_paths if path not in lr_images])
        dataset = list(zip(lr_images, hr_images))
        return dataset

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


class Set14(Dataset):
    def __init__(self, root, fmat='png', transform=None):
        ext = '*.%s' % fmat
        self.dataset = self.build_dataset(os.path.join(root, ext))
        self.transform = transform

    def build_dataset(self, root):
        image_paths = glob(root)
        lr_images = sorted([path for path in image_paths if 'LR' in path])
        hr_images = sorted([path for path in image_paths if path not in lr_images])
        dataset = list(zip(lr_images, hr_images))
        return dataset

    def __getitem__(self, idx):
        lres_img_path, hres_img_path = self.dataset[idx]
        lres_img = Image.open(lres_img_path)
        hres_img = Image.open(hres_img_path)
        sample = (lres_img, hres_img)

        if self.transform:
            sample = self.transform(sample)
        return sample


class BSD100(Dataset):
    def __init__(self, root, fmat='png', transform=None):
        ext = '*.%s' % fmat
        self.dataset = self.build_dataset(os.path.join(root, ext))
        self.transform = transform

    def build_dataset(self, root):
        image_paths = glob(root)
        lr_images = sorted([path for path in image_paths if 'LR' in path])
        hr_images = sorted([path for path in image_paths if path not in lr_images])
        dataset = list(zip(lr_images, hr_images))
        return dataset

    def __getitem__(self, idx):
        lres_img_path, hres_img_path = self.dataset[idx]
        lres_img = Image.open(lres_img_path)
        hres_img = Image.open(hres_img_path)
        sample = (lres_img, hres_img)

        if self.transform:
            sample = self.transform(sample)
        return sample


class Urban100(Dataset):
    def __init__(self, root, fmat='png', transform=None):
        ext = '*.%s' % fmat
        self.dataset = self.build_dataset(os.path.join(root, ext))
        self.transform = transform

    def build_dataset(self, root):
        image_paths = glob(root)
        lr_images = sorted([path for path in image_paths if 'LR' in path])
        hr_images = sorted([path for path in image_paths if 'HR' in path])
        dataset = list(zip(lr_images, hr_images))
        return dataset

    def __getitem__(self, idx):
        lres_img_path, hres_img_path = self.dataset[idx]
        lres_img = Image.open(lres_img_path)
        hres_img = Image.open(hres_img_path)
        sample = (lres_img, hres_img)

        if self.transform:
            sample = self.transform(sample)
        return sample


class PIRM(Dataset):
    def __init__(self, root, fmat='png', transform=None):
        ext = '*.%s' % fmat
        self.lr_root = os.path.join(root, '4x_downsampled', ext)
        self.hr_root = os.path.join(root, 'Original', ext)
        self.dataset = self.build_dataset(self.lr_root, self.hr_root)
        self.transform = transform

    def build_dataset(self, lr_root, hr_root):
        lr_images = sorted(glob(lr_root))
        hr_images = sorted(glob(hr_root))
        dataset = list(zip(lr_images, hr_images))
        return dataset

    def __getitem__(self, idx):
        lres_img_path, hres_img_path = self.dataset[idx]
        lres_img = Image.open(lres_img_path)
        hres_img = Image.open(hres_img_path)
        sample = (lres_img, hres_img)

        if self.transform:
            sample = self.transform(sample)
        return sample


class Div2K(Dataset):
    def __init__(self, root, fmat='png', transform=None):
        ext = '*.%s' % fmat
        self.lr_root = os.path.join(root, 'DIV2K_valid_LR_bicubic_X4', ext)
        self.hr_root = os.path.join(root, 'DIV2K_valid_HR', ext)
        self.dataset = self.build_dataset(self.lr_root, self.hr_root)
        self.transform = transform

    def build_dataset(self, lr_root, hr_root):
        lr_images = sorted(glob(lr_root))
        hr_images = sorted(glob(hr_root))
        dataset = list(zip(lr_images, hr_images))
        return dataset

    def __getitem__(self, idx):
        lres_img_path, hres_img_path = self.dataset[idx]
        lres_img = Image.open(lres_img_path)
        hres_img = Image.open(hres_img_path)
        sample = (lres_img, hres_img)

        if self.transform:
            sample = self.transform(sample)
        return sample
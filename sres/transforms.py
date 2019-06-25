import torch


class ToTensor:
    def __call__(self, sample):
        image1, image2 = sample
        sample = (torch.Tensor(image1), torch.Tensor(image2))
        return sample


class Transpose:
    def __call__(self, sample):
        image1, image2 = sample
        image1 = image1.transpose((2,0,1))
        image2 = image2.transpose((2,0,1))
        sample = (image1, image2)
        return sample

    
class Normalize:
    def __call__(self, sample):
        image1, image2 = sample
        image1 = 2 * (image1 - image1.min()) / (image1.max() - image1.min())-1
        image2 = 2 * (image2 - image2.min()) / (image2.max() - image2.min())-1
        sample = (image1, image2)
        return sample
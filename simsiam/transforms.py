# By Alexandre Hirsch

import torchvision.transforms as transforms
import torch

from simsiam.loader import GaussianBlur
#from simsiam.utils import show

class ColorJitter4(transforms.ColorJitter):
    def __call__(self, img: torch.tensor):
        img[:3] = super().__call__(img[:3])
        return img
    
class RandomGrayscale4(transforms.RandomGrayscale):
    def __call__(self, img: torch.tensor):
        img[:3] = super().__call__(img[:3])
        return img

class GaussianBlur4(GaussianBlur):
    T1 = transforms.ToPILImage()
    T2 = transforms.ToTensor()
    def __call__(self, img: torch.tensor):
        img[:3] = self.T2(super().__call__(self.T1(img[:3])))
        return img
        
class Normalize4(transforms.Normalize):
    def __call__(self, img: torch.tensor):
        img[:3] = super().__call__(img[:3])
        return img
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from data.stl10transforms import build_transform
from PIL import ImageFilter
import random
import torch
import os
from PIL import Image
import numpy as np

class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def get_mnist_data(
    mnist_root,
):
    data_train = datasets.MNIST(
        mnist_root, 
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.flatten(x))
        ])
    )
    
    class MNISTAugmentations:
        def __init__(self):
            self.augs = transforms.Compose([
                transforms.RandomChoice([
                    transforms.RandomAffine(degrees=30, translate=(.2,.2), scale=(0.75,1.25), shear=(-10,10,-10,10)),
                    transforms.RandomResizedCrop(28, scale=(0.5,1.2), ratio=(0.8,1.2)),
                ]),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: torch.flatten(x))
            ])
            self.no_augs = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: torch.flatten(x))
            ])

        def __call__(self, x):
            return self.no_augs(x), self.augs(x)
    
    data_train_augmented = datasets.MNIST(
        mnist_root, transform=MNISTAugmentations()
    )   
        
    data_test = datasets.MNIST(
        mnist_root,
        download=True,
        train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.flatten(x))
        ]),
    )
    return data_train, data_train_augmented, data_test

def get_fashion_mnist_data(
    data_root,
):
    data_train = datasets.FashionMNIST(
        data_root, 
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.flatten(x))
        ])
    )
    
    class FashionMNISTAugmentations:
        def __init__(self):
            self.augs = transforms.Compose([
                transforms.RandomResizedCrop(28, scale=(0.2,1.)),
                transforms.RandomHorizontalFlip(),
                GaussianBlur([0.1, 2.0]),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: torch.flatten(x))
            ])
            self.no_augs = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: torch.flatten(x))
            ])

        def __call__(self, x):
            return self.no_augs(x), self.augs(x)
    
    data_train_augmented = datasets.FashionMNIST(
        data_root, 
        transform=FashionMNISTAugmentations()
    )   
        
    data_test = datasets.FashionMNIST(
        data_root,
        download=True,
        train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.flatten(x))
        ]),
    )
    return data_train, data_train_augmented, data_test

def get_stl10_data(data_root):
    """
    Adapted from https://github.com/Yunfan-Li/Contrastive-Clustering
    """    
    data_train = datasets.STL10(
        data_root,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize(
                    (224, 224), interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
            ]
        )
    )
    
    data_train_augmented = datasets.STL10(
            data_root,
            split="train",
            download=True,
            transform=transforms.Compose(
            [
                transforms.Resize(
                    (224, 224), interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: (x,x))
            ]
        )
        )  
        
    data_test = datasets.STL10(
            data_root,
        download=True,
        split="test",
        transform=transforms.Compose(
            [
                transforms.Resize(
                    (224, 224), interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
            ]
        )
    )
    return data_train, data_train_augmented, data_test
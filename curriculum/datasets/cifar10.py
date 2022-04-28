# The code is developed based on 
# https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb#file-data_loader-py



import numpy as np

import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms

from .utils import Cutout



def get_cifar10_dataset(data_dir, valid_ratio=0.1,
                        shuffle=True, random_seed=43,
                        augment=True, cutout_length=0):

    train_dataset, valid_dataset = get_train_valid_dataset(
        data_dir, valid_ratio, shuffle, random_seed, augment, cutout_length
    )
    test_dataset = get_test_dataset(data_dir)

    return train_dataset, valid_dataset, test_dataset


def get_train_valid_dataset(data_dir, valid_ratio, 
                            shuffle, random_seed, 
                            augment, cutout_length):

    assert ((valid_ratio >= 0) and (valid_ratio <= 1)), \
        "[!] valid_size should be in the range [0, 1]."

    MEAN = [0.4914, 0.4822, 0.4465]
    STD =  [0.2023, 0.1994, 0.2010]

    transf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip()
    ] if augment else []
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]
    cutout = [Cutout(cutout_length)] \
      if cutout_length > 0 else []
    
    train_transform = transforms.Compose(transf + normalize + cutout)
    valid_transform = transforms.Compose(normalize)

    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=train_transform,
    )
    valid_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=valid_transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_ratio * num_train))
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_dataset = Subset(train_dataset, train_idx)
    valid_dataset = Subset(valid_dataset, valid_idx)

    return train_dataset, valid_dataset


def get_test_dataset(data_dir):

    MEAN = [0.485, 0.456, 0.406]
    STD =  [0.229, 0.224, 0.225]

    test_normalize = [
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]

    test_transform = transforms.Compose(test_normalize)

    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False,
        download=True, transform=test_transform,
    )

    return test_dataset
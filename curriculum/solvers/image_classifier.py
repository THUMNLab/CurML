import os
import time

import random
import numpy as np

import torch
import torchvision


from ..datasets import get_dataset
from ..backbones import get_model
from ..utils import get_logger



class ImageClassifier:
    def __init__(self, data_name, model_name):

        self._init_dataloader(data_name)
        self._init_model(model_name)


    def _init_dataloader(self, data_name):
        train_dataset, valid_dataset, test_dataset = \
            get_dataset('./data', self.data_name)

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=128, shuffle=True,
            num_workers=2, pin_memory=True,
        )
        self.valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=128, shuffle=False,
            num_workers=2, pin_memory=True,
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=128, shuffle=False,
            num_workers=2, pin_memory=True,
        )


    def _init_model(self, model_name):
        self.model = get_model(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

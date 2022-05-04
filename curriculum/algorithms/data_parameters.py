import numpy as np
import torch

from .base import BaseTrainer, BaseCL
from .utils import SparseSGD


class DataParameters(BaseCL):
    def __init__(self, class_size, device):
        super(DataParameters, self).__init__()

        self.name = 'dataparameters'
        self.data_weights = None
        self.data_optimizer = None
        self.class_weights = None
        self.class_optimizer = None

        self.class_size = class_size
        self.device = device


    def data_curriculum(self, loader):
        loader = super().data_curriculum(loader)

        if self.data_optimizer is None:
            self.data_weights = torch.tensor(
                np.ones(self.data_size) * np.log(1.0),
                dtype=torch.float32, requires_grad=True, device=self.device
            )
            self.data_optimizer = SparseSGD([self.data_weights], 
                lr=0.1, momentum=0.9, skip_update_zero_grad=True
            )
            self.data_optimizer.zero_grad()
        if self.class_optimizer is None:
            self.class_weights = torch.tensor(
                np.ones(self.class_size) * np.log(1.0),
                dtype=torch.float32, requires_grad=True, device=self.device
            )
            self.class_optimizer = SparseSGD([self.class_weights], 
                lr=0.1, momentum=0.9, skip_update_zero_grad=True
            )
            self.class_optimizer.zero_grad()

        return loader


    def loss_curriculum(self, criterion, outputs, labels, indices):
        # update last batch
        self.data_optimizer.step()
        self.class_optimizer.step()

        self.data_weights.data.clamp_(min=np.log(1/20), max=np.log(20))
        self.class_weights.data.clamp_(min=np.log(1/20), max=np.log(20))

        # calculate current batch
        self.data_optimizer.zero_grad()
        self.class_optimizer.zero_grad()

        data_weights = self.data_weights[indices]
        class_weights = self.class_weights[labels]
        sigma = torch.exp(data_weights) + torch.exp(class_weights)

        loss = criterion(outputs / sigma.view(-1, 1), labels)
        return torch.mean(loss)


class DataParametersTrainer(BaseTrainer):
    def __init__(self, data_name, net_name, 
                 device_name, random_seed):
        
        if data_name in ['cifar10']:
            cl = DataParameters(
                class_size=10, device=torch.device(device_name),
            )
        else:
            raise NotImplementedError()
        
        super(DataParametersTrainer, self).__init__(
            data_name, net_name, device_name, random_seed, cl
        )





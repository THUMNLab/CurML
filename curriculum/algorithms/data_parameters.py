import numpy as np
import torch

from .base import BaseTrainer, BaseCL
from .utils import SparseSGD



class DataParameters(BaseCL):
    def __init__(self, class_size, init_class_param, lr_class_param, wd_class_param, 
                 init_data_param, lr_data_param, wd_data_param):
        super(DataParameters, self).__init__()

        self.name = 'data_parameters'

        self.class_size = class_size
        self.init_class_param = init_class_param
        self.lr_class_param = lr_class_param
        self.wd_class_param = wd_class_param
        self.init_data_param = init_data_param
        self.lr_data_param = lr_data_param
        self.wd_data_param = wd_data_param


    def model_prepare(self, net, device, epochs, 
                      criterion, optimizer, lr_scheduler):
        self.device = device

        self.data_weights = torch.tensor(
            np.ones(self.data_size) * np.log(self.lr_data_param),
            dtype=torch.float32, requires_grad=True, device=self.device
        )
        self.data_optimizer = SparseSGD([self.data_weights], 
            lr=self.lr_data_param, momentum=0.9, skip_update_zero_grad=True
        )
        self.data_optimizer.zero_grad()

        self.class_weights = torch.tensor(
            np.ones(self.class_size) * np.log(self.lr_class_param),
            dtype=torch.float32, requires_grad=True, device=self.device
        )
        self.class_optimizer = SparseSGD([self.class_weights], 
            lr=self.lr_class_param, momentum=0.9, skip_update_zero_grad=True
        )
        self.class_optimizer.zero_grad()


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

        loss = criterion(outputs / sigma.view(-1, 1), labels)           \
             + (0.5 * self.wd_data_param * data_weights ** 2).sum()     \
             + (0.5 * self.wd_class_param * class_weights ** 2).sum()
        return torch.mean(loss)


class DataParametersTrainer(BaseTrainer):
    def __init__(self, data_name, net_name, device_name, num_epochs, random_seed,
                 init_class_param, lr_class_param, wd_class_param, init_data_param, lr_data_param, wd_data_param):
        
        if data_name.startswith('cifar10'):
            cl = DataParameters(10, init_class_param, lr_class_param, wd_class_param, init_data_param, lr_data_param, wd_data_param)
        else:
            raise NotImplementedError()
        
        super(DataParametersTrainer, self).__init__(
            data_name, net_name, device_name, num_epochs, random_seed, cl)





import numpy as np
import torch

from .base import BaseTrainer, BaseCL
from .utils import SparseSGD



class DataParameters(BaseCL):
    """
    
    Data parameters: A new family of parameters for learning a differentiable curriculum. https://proceedings.neurips.cc/paper/2019/file/926ffc0ca56636b9e73c565cf994ea5a-Paper.pdf
    """
    def __init__(self, class_size):
        super(DataParameters, self).__init__()

        self.name = 'dataparameters'

        self.class_size = class_size


    def model_prepare(self, net, device, epochs, 
                      criterion, optimizer, lr_scheduler):
        self.device = device

        self.data_weights = torch.tensor(
            np.ones(self.data_size) * np.log(1.0),
            dtype=torch.float32, requires_grad=True, device=self.device
        )
        self.data_optimizer = SparseSGD([self.data_weights], 
            lr=0.1, momentum=0.9, skip_update_zero_grad=True
        )
        self.data_optimizer.zero_grad()

        self.class_weights = torch.tensor(
            np.ones(self.class_size) * np.log(1.0),
            dtype=torch.float32, requires_grad=True, device=self.device
        )
        self.class_optimizer = SparseSGD([self.class_weights], 
            lr=0.1, momentum=0.9, skip_update_zero_grad=True
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

        loss = criterion(outputs / sigma.view(-1, 1), labels)
        return torch.mean(loss)


class DataParametersTrainer(BaseTrainer):
    def __init__(self, data_name, net_name, device_name, num_epochs, random_seed):
        
        if data_name.startswith('cifar10'):
            cl = DataParameters(10)
        else:
            raise NotImplementedError()
        
        super(DataParametersTrainer, self).__init__(
            data_name, net_name, device_name, num_epochs, random_seed, cl)





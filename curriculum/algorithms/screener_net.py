from .base import BaseTrainer, BaseCL
import copy
from torch.autograd import Variable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.sgd import SGD

class VNet(nn.Module):
    def __init__(self, input, hidden):
        super(VNet, self).__init__()
        self.linear1 = nn.Linear(input, hidden)

    def forward(self, x):
        x = self.linear1(x)
        return torch.sigmoid(x)

def set_parameter(current_module, name, parameters):
        if '.' in name:
            name_split = name.split('.')
            module_name = name_split[0]
            rest_name = '.'.join(name_split[1:])
            for children_name, children in current_module.named_children():
                if module_name == children_name:
                    set_parameter(children, rest_name, parameters)
                    break
        else:
            current_module._parameters[name] = parameters

class ScreenerNet(BaseCL):
    def __init__(self, ):
        super(ScreenerNet, self).__init__()

        self.name = 'screenernet'

        self.catnum = 10
        self.lr = 1e-3       
        
    def data_prepare(self, loader):
        self.dataset = loader.dataset
        self.data_size = len(self.dataset)
        self.batch_size = loader.batch_size
        self.n_batches = (self.data_size - 1) // self.batch_size + 1
        self.trainData = loader
        self.iter = iter(self.trainData)

    def model_prepare(self, net, device, epochs, criterion, optimizer, lr_scheduler):
        self.model = net.to(device)
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        
        self.vnet_ = copy.deepcopy(self.model)
        self.linear = VNet(self.catnum, 1).to(self.device)
        self.optimizer1 = SGD(self.vnet_.parameters(), lr=self.lr, weight_decay=0.01)
        self.optimizer2 = SGD(self.linear.parameters(), lr=self.lr, weight_decay=0.01)

    def data_curriculum(self, loader):
        self.model.train()
        self.vnet_.train()
        self.linear.train()   
        try:
            temp = next(self.iter)
        except StopIteration:
            self.trainData = DataLoader(self.trainData.dataset, self.batch_size, shuffle=True)
            self.iter = iter(self.trainData)
            temp = next(self.iter)
        image, labels = temp
        image = image.to(self.device)
        labels = labels.to(self.device)
        l = self.criterion(self.model(image), labels)
        w = self.linear(self.vnet_(image))
        L = Variable(torch.zeros(1), requires_grad=True).to(self.device)
        for i, j in zip(l, w):
            L = L+ (1-j)*(1-j)*i + j*j*max(1-i, 0)
        L.backward()
        self.optimizer1.step()
        self.optimizer2.step()
        w_tilde = self.linear(self.vnet_(image))
        norm_c = torch.sum(w_tilde)

        if norm_c != 0:
            w = w_tilde / norm_c
        else:
            w = w_tilde
        w = w * self.batch_size

        return [[image, labels, w]]

class ScreenerNetTrainer(BaseTrainer):
    def __init__(self, data_name, net_name, device_name, num_epochs, random_seed):
        
        cl = ScreenerNet()

        super(ScreenerNetTrainer, self).__init__(
            data_name, net_name, device_name, num_epochs, random_seed, cl
        )
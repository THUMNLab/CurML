from .base import BaseTrainer, BaseCL
import copy
from torch.autograd import Variable

import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
from torch.optim.sgd import SGD

from .utils import VNet_, set_parameter



class ScreenerNet(BaseCL):
    """ScreenerNet CL Algorithm. 
    
    Screenernet: Learning self-paced curriculum for deep neural networks. https://arxiv.org/pdf/1801.00904
    """
    def __init__(self, ):
        super(ScreenerNet, self).__init__()

        self.name = 'screener_net'

        self.catnum = 10
        self.lr = 1e-3       
        

    def data_prepare(self, loader):
        super().data_prepare(loader)

        self.trainData = DataLoader(self.dataset, self.batch_size, shuffle=True)
        self.iter = iter(self.trainData)
        self.weights = torch.zeros(self.data_size)


    def model_prepare(self, net, device, epochs, criterion, optimizer, lr_scheduler):
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.model = net.to(device)
        self.weights = self.weights.to(self.device)
        
        self.vnet_ = copy.deepcopy(self.model)
        self.linear = VNet_(self.catnum, 1).to(self.device)
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
        image, labels, indices = temp
        image = image.to(self.device)
        labels = labels.to(self.device)
        indices = indices.to(self.device)
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
        self.weights[indices] = w.view(1, -1).detach()
        return [[image, labels, indices]]


    def loss_curriculum(self, criterion, outputs, labels, indices):
        return torch.mean(criterion(outputs, labels) * self.weights[indices])



class ScreenerNetTrainer(BaseTrainer):
    def __init__(self, data_name, net_name, device_name, num_epochs, random_seed):
        
        cl = ScreenerNet()

        super(ScreenerNetTrainer, self).__init__(
            data_name, net_name, device_name, num_epochs, random_seed, cl)
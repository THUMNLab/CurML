from .base import BaseTrainer, BaseCL

import copy

import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
from torch.optim.sgd import SGD
import numpy as np

from .utils import VNet_, set_parameter



class DDS(BaseCL):
    """
    
    Optimizing data usage via differentiable rewards. http://proceedings.mlr.press/v119/wang20p/wang20p.pdf
    """
    def __init__(self, catnum, epsilon, lr):
        super(DDS, self).__init__()

        self.name = 'dds'
        self.catnum = catnum
        self.epsilon = epsilon
        self.lr = lr   


    def randomSplit(self):
        """split data into train and validation data by proportion 9:1"""
        sample_size = self.data_size//10
        temp = np.array(range(self.data_size))
        np.random.shuffle(temp)
        valid_index = temp[:sample_size]
        train_index = temp[sample_size:]
        self.validationData = DataLoader(Subset(self.dataset, valid_index), self.batch_size, shuffle = False)
        self.trainData = DataLoader(Subset(self.dataset, train_index), self.batch_size, shuffle = True)
        self.iter1 = iter(self.trainData)
        self.iter2 = iter(self.validationData)

        self.weights = torch.zeros(self.data_size)
       

    def data_prepare(self, loader):
        super().data_prepare(loader)
        
        self.randomSplit()


    def model_prepare(self, net, device, epochs, criterion, optimizer, lr_scheduler):
        # super().model_prepare(net, device, epochs, criterion, optimizer, lr_scheduler)
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.model = net.to(device)
        self.weights = self.weights.to(self.device)

        self.last_net = copy.deepcopy(self.model)
        self.vnet_ = copy.deepcopy(self.model)
        self.linear = VNet_(self.catnum, 1).to(self.device)
        self.image, self.label, self.indices = next(self.iter1)


    def data_curriculum(self, loader):
        self.model.train()
        self.vnet_.train()
        self.linear.train()
        try:
            temp2 = next(self.iter2)
        except StopIteration:
            self.validationData = DataLoader(self.validationData.dataset, self.batch_size, shuffle=True)
            self.iter2 = iter(self.validationData)
            temp2 = next(self.iter2)

        image, labels, indices = self.image, self.label, self.indices
        image = image.to(self.device)
        labels = labels.to(self.device)
        indices = indices.to(self.device)

        out = self.last_net(image)
#        self.last_net.zero_grad()
        with torch.no_grad():
            loss = self.criterion(out, labels)

        image2, labels2, indices2 = temp2
        image2 = image2.to(self.device)
        labels2 = labels2.to(self.device)
        out2 = self.model(image2)
        loss2 = self.criterion(out2, labels2)
        totalloss2 = torch.mean(loss2)
        self.model.zero_grad()
        grad = torch.autograd.grad(totalloss2, self.model.parameters(), create_graph=True, retain_graph=True)

        for (name, parameter), j in zip(self.last_net.named_parameters(), grad):
            parameter.detach_()
            set_parameter(self.last_net, name, parameter.add(j, alpha = -self.epsilon))
        with torch.no_grad():
            loss3 = self.criterion(self.last_net(image), labels)
        r = (loss3 -loss)/self.epsilon
        #print(r)
        out3 = self.vnet_(image)
        out4 = out3.reshape(out3.size() , -1)
        out5 = self.linear(out4)
        out5_norm = torch.sum(out5)
        if out5_norm != 0:
            out5_ = out5/out5_norm
        else:
            out5_ = out5
        L = torch.sum(r * torch.log(out5_) )

        grad1 = torch.autograd.grad(L, self.linear.parameters(), create_graph=True, retain_graph=True)
        grad2 = torch.autograd.grad(L, self.vnet_.parameters())
        #print(grad1)
        #print(grad2)
        for (name, parameter), j in zip(self.linear.named_parameters(), grad1):
            set_parameter(self.linear, name, parameter.add(j, alpha = -self.lr))
        for (name, parameter), j in zip(self.vnet_.named_parameters(), grad2):
            set_parameter(self.vnet_, name, parameter.add(j, alpha = -self.lr))
        del grad1
        del grad2
        del self.last_net
        self.last_net = copy.deepcopy(self.model)

        try:
            temp = next(self.iter1)
        except StopIteration:
            self.trainData = DataLoader(self.trainData.dataset, self.batch_size, shuffle=True)
            self.iter1 = iter(self.trainData)
            temp = next(self.iter1)
        a, b, i = temp
        self.image = copy.deepcopy(a)
        self.label = copy.deepcopy(b)

        a = a.to(self.device)
        #print(a)
        self.vnet_.eval()
        self.linear.eval()
        z = self.vnet_(a)
        #print(z)
        w = self.linear(z)    
        w_norm = torch.sum(w)
        if w_norm != 0:
            w_ = w / w_norm
        else :
            w_ = w
        #print(w_)
#        c = Variable(a, requires_grad=False)
#        d = Variable(b, requires_grad=False)
#        e = Variable(w_, requires_grad=False)
        a.detach_()
        b.detach_()
        w_.detach_()
#        del a
#        del b
        self.weights[i] = w.view(1, -1).detach()
        return [[a, b, i]]


    def loss_curriculum(self, criterion, outputs, labels, indices):
        return torch.mean(criterion(outputs, labels) * self.weights[indices])



class DDSTrainer(BaseTrainer):
    def __init__(self, data_name, net_name, 
                 device_name, num_epochs, random_seed,
                 catnum, epsilon, lr):
        
        cl = DDS(catnum, epsilon, lr)

        super(DDSTrainer, self).__init__(
            data_name, net_name, device_name, num_epochs, random_seed, cl)
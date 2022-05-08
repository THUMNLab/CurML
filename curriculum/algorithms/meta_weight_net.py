from .base import BaseTrainer, BaseCL

import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.sgd import SGD
import numpy as np


class VNet(nn.Module):
    def __init__(self, input, hidden, output):
        super(VNet, self).__init__()
        self.linear1 = nn.Linear(input, hidden)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(hidden, output)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        out = self.linear2(x)
        return torch.sigmoid(out)


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



class MetaWeightNet(BaseCL):
    def __init__(self, ):
        super(MetaWeightNet, self).__init__()

        self.name = 'metaweightnet'

    def randomSplit(self):
        """split data into train and validation data by proportion 9:1"""
        sample_size = self.data_size//10
        temp = np.array(range(self.data_size))
        np.random.shuffle(temp)
        valid_index = temp[:sample_size]
        train_index = temp[sample_size:]
        self.validationData = DataLoader(torch.utils.data.Subset(self.dataset, valid_index), self.batch_size, shuffle = False)
        self.trainData = DataLoader(torch.utils.data.Subset(self.dataset, train_index), self.batch_size, shuffle = True)
        self.iter = iter(self.trainData)
        self.iter2 = iter(self.validationData)
       
    def data_prepare(self, loader):
        self.dataset = loader.dataset
        self.data_size = len(self.dataset)
        self.batch_size = loader.batch_size
        self.n_batches = (self.data_size - 1) // self.batch_size + 1
        
        self.randomSplit()

    def model_prepare(self, net, device, epochs, criterion, optimizer, lr_scheduler):
        # super().model_prepare(net, device, epochs, criterion, optimizer, lr_scheduler)
        self.model = net.to(device)
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.vnet = VNet(1, 100, 1).to(self.device)

    def data_curriculum(self, loader):    
        self.model.train()
        try:
            temp = next(self.iter)
        except StopIteration:
            self.trainData = DataLoader(self.trainData.dataset, self.batch_size, shuffle=True)
            self.iter = iter(self.trainData)
            temp = next(self.iter)
        try:
            temp2 = next(self.iter2)
        except StopIteration:
            self.validationData = DataLoader(self.validationData.dataset, self.batch_size, shuffle=True)
            self.iter2 = iter(self.validationData)
            temp2 = next(self.iter2)
        image, labels = temp
        image = image.to(self.device)
        labels = labels.to(self.device)
        pseudonet = copy.deepcopy(self.model)
        out = pseudonet(image)
        loss = self.criterion(out, labels)
        loss = loss.reshape(-1, 1)
        w_w = self.vnet(loss)
        w_norm = torch.sum(w_w)
        if w_norm != 0:
            eps = w_w / w_norm
        else :
            eps = w_w

        lr = 0.001
        totalloss1 = torch.sum(eps * loss)

        grad = torch.autograd.grad(totalloss1, pseudonet.parameters(), create_graph=True, retain_graph=True)

        for (name, parameter), j in zip(pseudonet.named_parameters(), grad):
            parameter.detach_()
            set_parameter(pseudonet, name, parameter.add(j, alpha = -lr))


        totalloss2 = 0
        image2, label2 = temp2

        image2 = image2.to(self.device)
        label2 = label2.to(self.device)
        out2 = pseudonet(image2)
        loss2 = self.criterion(out2, label2)
        totalloss2 += torch.sum(loss2)

        grad_eps = torch.autograd.grad(totalloss2, self.vnet.parameters())
        for (name, parameter), j in zip(self.vnet.named_parameters(), grad_eps):
            set_parameter(self.vnet, name, parameter.add(j, alpha = -lr))
        w_tilde = self.vnet(loss)
        norm_c = torch.sum(w_tilde)

        if norm_c != 0:
            w = w_tilde / norm_c
        else:
            w = w_tilde
        w = w * self.batch_size

        return [[image, labels, w]]
    



class MetaWeightNetTrainer(BaseTrainer):
    def __init__(self, data_name, net_name, device_name, num_epochs, random_seed):
        
        cl = MetaWeightNet()

        super(MetaWeightNetTrainer, self).__init__(
            data_name, net_name, device_name, num_epochs, random_seed, cl
        )

class VNet_(nn.Module):
    def __init__(self, input, hidden):
        super(VNet_, self).__init__()
        self.linear1 = nn.Linear(input, hidden)

    def forward(self, x):
        x = self.linear1(x)
        return torch.sigmoid(x)

class MetaWeightNet_2(BaseCL):
    def __init__(self, ):
        super(MetaWeightNet_2, self).__init__()

        self.name = 'metaweightnet_2'
        self.catnum = 10
        self.epsilon = 1e-3
        self.lr = 1e-4     

    def randomSplit(self):
        """split data into train and validation data by proportion 9:1"""
        sample_size = self.data_size//10
        temp = np.array(range(self.data_size))
        np.random.shuffle(temp)
        valid_index = temp[:sample_size]
        train_index = temp[sample_size:]
        self.validationData = DataLoader(torch.utils.data.Subset(self.dataset, valid_index), self.batch_size, shuffle = False)
        self.trainData = DataLoader(torch.utils.data.Subset(self.dataset, train_index), self.batch_size, shuffle = True)
        self.iter1 = iter(self.trainData)
        self.iter2 = iter(self.validationData)
       
    def data_prepare(self, loader):
        self.dataset = loader.dataset
        self.data_size = len(self.dataset)
        self.batch_size = loader.batch_size
        self.n_batches = (self.data_size - 1) // self.batch_size + 1
        
        self.randomSplit()

    def model_prepare(self, net, device, epochs, criterion, optimizer, lr_scheduler):
        # super().model_prepare(net, device, epochs, criterion, optimizer, lr_scheduler)
        self.model = net.to(device)
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.last_net = copy.deepcopy(self.model)
        self.vnet_ = copy.deepcopy(self.model)
        self.linear = VNet_(self.catnum, 1).to(self.device)
        self.image, self.label = next(self.iter1)

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

        image, labels = self.image, self.label
        image = image.to(self.device)
        labels = labels.to(self.device)


        out = self.last_net(image)
#        self.last_net.zero_grad()
        with torch.no_grad():
            loss = self.criterion(out, labels)

        image2, labels2 = temp2
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
        a, b = temp
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
        return [[a, b, w_]] 

class MetaWeightNetTrainer_2(BaseTrainer):
    def __init__(self, data_name, net_name, device_name, num_epochs, random_seed):
        
        cl = MetaWeightNet_2()

        super(MetaWeightNetTrainer_2, self).__init__(
            data_name, net_name, device_name, num_epochs, random_seed, cl
        )
from .base import BaseTrainer, BaseCL
import torch
from torch.utils.data import Subset, DataLoader
import numpy as np
import torch.nn as nn
from torch.optim.sgd import SGD
import copy

from .utils import set_parameter



class MetaReweight(BaseCL):
    """Meta Reweight CL Algorithm.
    
    Learning to reweight examples for robust deep learning. http://proceedings.mlr.press/v80/ren18a/ren18a.pdf
    """
    def __init__(self, ):
        super(MetaReweight, self).__init__()
        self.name = 'meta_reweight'
    

    def randomSplit(self):
        """split data into train and validation data by proportion 9:1"""
        sample_size = self.data_size//10
        temp = np.array(range(self.data_size))
        np.random.shuffle(temp)
        valid_index = temp[:sample_size]
        train_index = temp[sample_size:]
        self.validationData = DataLoader(Subset(self.dataset, valid_index), self.batch_size, shuffle = False)
        self.trainData = DataLoader(Subset(self.dataset, train_index), self.batch_size, shuffle = True)
        self.iter = iter(self.trainData)
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
        

    def data_curriculum(self, loader):
        self.model.train()
        
        try:
            temp = next(self.iter)
        except StopIteration:
            # self.trainData = DataLoader(self.trainData.dataset, self.batch_size, shuffle=True)
            self.iter = iter(self.trainData)
            temp = next(self.iter)
        
        try:
            temp2 = next(self.iter2)
        except StopIteration:
            # self.validationData = DataLoader(self.validationData.dataset, self.batch_size, shuffle=True)
            self.iter2 = iter(self.validationData)
            temp2 = next(self.iter2)

        image, labels, indices = temp
        image = image.to(self.device)
        labels = labels.to(self.device)
        indices = indices.to(self.device)
        pseudonet = copy.deepcopy(self.model)
        out = pseudonet(image)
        loss = self.criterion(out, labels)
        eps = nn.Parameter(torch.zeros(loss.size())).to(self.device)
        lr = 0.01
        totalloss1 = torch.sum(eps * loss)

        grad = torch.autograd.grad(totalloss1, pseudonet.parameters(), create_graph=True, retain_graph=True)

        for (name, parameter), j in zip(pseudonet.named_parameters(), grad):
            parameter.detach_()
            set_parameter(pseudonet, name, parameter.add(j, alpha = -lr))


        totalloss2 = 0
        image2, label2, indices2 = temp2

        image2 = image2.to(self.device)
        label2 = label2.to(self.device)
        out2 = pseudonet(image2)
        loss2 = self.criterion(out2, label2)
        totalloss2 += torch.sum(loss2)

        grad_eps = torch.autograd.grad(totalloss2, eps)
        w_tilde = torch.clamp(-grad_eps[0], min=0)
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



class MetaReweightTrainer(BaseTrainer):
    def __init__(self, data_name, net_name, device_name, num_epochs, random_seed):
        
        cl = MetaReweight()

        super(MetaReweightTrainer, self).__init__(
            data_name, net_name, device_name, num_epochs, random_seed, cl)
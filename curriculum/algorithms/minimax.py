import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader

from .base import BaseTrainer, BaseCL



class Minimax(BaseCL):
    """Minimax CL Algorithm.
    
    Minimax curriculum learning: Machine teaching with desirable difficulties and scheduled diversity. https://openreview.net/pdf?id=BywyFQlAW
    """
    def __init__(self, schedule_epoch, warm_epoch, lam, minlam, gamma, delta,
                 initial_size, fe_alpha, fe_beta, fe_gamma, fe_lambda,
                 fe_entropy, fe_gsrow, fe_central_op, fe_central_min, 
                 fe_central_sum, num_classes):
        super(Minimax, self).__init__()

        self.name = 'minimax'

        self.epoch = 0
        self.schedule_epoch = schedule_epoch
        self.warm_epoch = warm_epoch
        self.lam = lam
        self.minlam = minlam
        self.gamma = gamma
        self.cnt = 0
        self.initial_size = initial_size
        self.delta = delta
        self.num_classes = num_classes
        self.fe_alpha = fe_alpha
        self.fe_beta = fe_beta
        self.fe_gamma = fe_gamma
        self.fe_lambda = fe_lambda
        self.fe_entropy = fe_entropy
        self.fe_gsrow = fe_gsrow
        self.fe_central_op = fe_central_op
        self.fe_central_min = fe_central_min
        self.fe_central_sum = fe_central_sum
    

    def data_prepare(self, loader):
        self.dataloader = loader
        self.dataset = self.CLDataset(loader.dataset)
        self.data_size = len(self.dataset)
        self.batch_size = loader.batch_size
        self.n_batches = (self.data_size - 1) // self.batch_size + 1
        if self.initial_size is None:
            self.siz = 0.1 * self.data_size
        else:
            self.siz = self.initial_size * self.data_size
        
        self.model = self._Resnet18(num_classes=self.num_classes)
        self.loss = np.zeros(self.data_size)
        self.features = np.zeros(self.data_size)
        self.centrality = np.zeros(self.data_size)
        self.train_set = np.arange(self.data_size)


    def model_prepare(self, net, device, epochs, criterion, optimizer, lr_scheduler):
        self.total_epoch = epochs
        if self.delta is None:
            self.delta = int((self.data_size - self.siz) / (int(self.total_epoch / self.schedule_epoch)))
        else:
            self.delta = self.delta * self.data_size
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.total_epoch)
        self.device = device
        self.model.to(self.device)


    def data_curriculum(self, loader):
        if self.epoch % self.schedule_epoch == 0:
            if self.epoch != 0:
                self.lam = max(self.lam * (1 - self.gamma), self.minlam)
                self.siz = min(self.siz + self.delta, self.data_size)
            # at the begining of each episode, train a neural network for few epochs to calculate the features used in the following MCL steps
            dataloader_ = DataLoader(self.dataset, self.batch_size, shuffle=False)
            self._pretrain(dataloader_)
            self.features = self._pretest(dataloader_)
            self.features = np.clip(self.features, 1e-10, 1e10)
            self.centrality = self._compute_centrality(self.features, self.fe_alpha, self.fe_beta, self.fe_gamma, self.fe_lambda, \
                                                    self.fe_gsrow, self.fe_entropy, self.fe_central_op, self.fe_central_min, self.fe_central_sum)

        if self.epoch < self.warm_epoch:
            dataloader = DataLoader(self.dataset, self.batch_size, shuffle=False)
        else:
            pro = self.loss + self.lam * self.centrality
            pro = pro / np.sum(pro)
            self.train_set = np.random.choice(self.data_size, int(self.siz), p=pro, replace=False)
            dataset = Subset(self.dataset, self.train_set)
            dataloader = DataLoader(dataset, self.batch_size, shuffle=False)

        self.epoch += 1
        self.cnt = 0
        return dataloader
    

    def loss_curriculum(self, criterion, outputs, labels, indices):
        losses = criterion(outputs, labels)
        for loss in losses:
            self.loss[self.train_set[self.cnt]] = loss
            self.cnt += 1
        return torch.mean(losses)
    

    def _pretrain(self, dataloader):
        self.model.train()
        for step, data in enumerate(dataloader):
            inputs = data[0].to(self.device)
            labels = data[1].to(self.device)
            self.optimizer.zero_grad()
            outputs, _ = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
        self.scheduler.step()
    

    def _pretest(self, dataloader):
        all_feature = np.array([])
        self.model.eval()
        for step, data in enumerate(dataloader):
            inputs = data[0].to(self.device)
            with torch.no_grad():
                _, feature = self.model(inputs)
            all_feature = np.append(all_feature, feature.cpu())
        all_feature = all_feature.reshape(int(self.data_size), int(len(all_feature) / self.data_size))
        return all_feature
    

    def _entropy(self, labels, base=None):
        num_labels = len(labels)
        value, count = np.unique(labels, return_counts=True)
        prob = count / num_labels
        num_classes = np.count_nonzero(prob)
        if num_labels <= 1 or num_classes <= 1:
            return 1
        entro = 0
        if base == None:
            base = np.e 
        for iter in prob:
            entro -= iter * math.log(iter, base)
        return entro
    

    def _swarp(self, feature, alpha, beta):
        swarp_epsilon = 1e-30
        nonzero_indice = (feature > swarp_epsilon)
        if np.any(nonzero_indice) > 0:
            feature[nonzero_indice] = 1.0 / (1 + (feature[nonzero_indice] ** (1 / np.log2(beta)) - 1) ** alpha)
        return feature


    def _compute_centrality(self, feature, fe_alpha, fe_beta, fe_gamma, fe_lambda, fe_gsmin, fe_entropy, fe_central_op, fe_central_min, fe_central_sum):
        # First process the feature matrix, using gamma correction
        if fe_gamma != 1.0 or fe_alpha != 1.0 or fe_beta != 0.5:
            if fe_gsmin == True:
                feature_min = feature.min(axis=0)
                feature_max = feature.max(axis=0) + 1e-5
                feature = feature - feature_min
                feature = feature / feature_max
            else:
                feature_max = feature.max(axis=0)
                feature = feature / feature_max
        
        if fe_gamma != 1.0:
            if fe_alpha != 1.0 or fe_beta != 0.5:
                feature = self._swarp(feature, fe_alpha, fe_beta) ** fe_gamma
            else:
                feature = feature ** fe_gamma
        feature = feature * feature_max

        # Then compute the centrality
        centrality = None
        if fe_lambda < 1.0:
            centrality = np.zeros(feature.shape[0])
            if fe_entropy is True:
                max_entropy = np.log2(feature.shape[1])
                for index in range(feature.shape[0]):
                    centrality[index] = max_entropy - self._entropy(feature[index].T)
                centrality = centrality / np.sum(centrality)
            else:
                if fe_central_op is True:
                    centrality = np.sum(feature.dot(feature.transpose()), axis=1)
                    centrality = centrality / np.sum(centrality)
                elif fe_central_min is True:
                    centrality = np.min(feature, axis=1)
                    centrality = centrality / np.sum(centrality)
                elif fe_central_sum is True:
                    centrality = np.sum(feature, axis=1)
                    centrality = centrality / np.sum(centrality)
        return centrality
        

    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, in_planes, planes, stride=1):
            super().__init__()
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != self.expansion*planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes,
                            kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = F.relu(out)
            return out
    

    class Resnet(nn.Module):
        def __init__(self, block, num_blocks, num_classes=10):
            super().__init__()
            self.in_planes = 64

            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
            self.linear = nn.Linear(512*block.expansion, num_classes)

        def _make_layer(self, block, planes, num_blocks, stride):
            strides = [stride] + [1]*(num_blocks-1)
            layers = []
            for stride in strides:
                layers.append(block(self.in_planes, planes, stride))
                self.in_planes = planes * block.expansion
            return nn.Sequential(*layers)

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            feature = out
            out = self.linear(out)
            return out, feature
    
    def _Resnet18(self, num_classes):
        return self.Resnet(self.BasicBlock, [2, 2, 2, 2], num_classes)



class MinimaxTrainer(BaseTrainer):
    def __init__(self, data_name, net_name, device_name, num_epochs, random_seed,
                 schedule_epoch, warm_epoch, lam, minlam, gamma, delta,
                 initial_size, fe_alpha, fe_beta, fe_gamma, fe_lambda,
                 fe_entropy, fe_gsrow, fe_central_op, fe_central_min, fe_central_sum,
                 num_classes):
        
        cl = Minimax(schedule_epoch, warm_epoch, lam, minlam, gamma, delta,
                 initial_size, fe_alpha, fe_beta, fe_gamma, fe_lambda,
                 fe_entropy, fe_gsrow, fe_central_op, fe_central_min, fe_central_sum,
                 num_classes)

        super(MinimaxTrainer, self).__init__(
            data_name, net_name, device_name, num_epochs, random_seed, cl)

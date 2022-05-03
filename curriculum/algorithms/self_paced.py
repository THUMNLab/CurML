import math
import torch
from torch.utils.data import Subset, DataLoader

from .base import BaseTrainer, BaseCL



class SelfPaced(BaseCL):
    def __init__(self, start_rate, grow_epochs, grow_fn, 
                 weight_fn, criterion, device):
        super(SelfPaced, self).__init__()

        self.name = 'selfpaced'
        self.start_rate = start_rate
        self.grow_epochs = grow_epochs
        self.grow_fn = grow_fn
        self.weight_fn = weight_fn
        self.criterion = criterion
        self.device = device


    def model_curriculum(self, net):
        self.net = net
        return net


    def data_curriculum(self, loader):
        super().data_curriculum(loader)

        data_rate = min(1.0, self._subset_grow())
        data_size = int(self.data_size * data_rate)

        data_loss = self._loss_measure()
        data_indices = torch.argsort(data_loss)[:data_size]
        data_threshold = data_loss[data_indices[-1]]

        if self.weight_fn == 'hard':
            dataset = Subset(self.dataset, tuple(range(data_size)))
        else:
            weights = self._loss_reweight(data_loss, data_threshold)
            dataset = self.dataset.set_weights(weights)
        return DataLoader(dataset, self.batch_size, shuffle=True)


    def _subset_grow(self):
        if self.grow_fn == 'linear':
            return self.start_rate + (1.0 - self.start_rate) / self.grow_epochs * self.epoch
        elif self.grow_fn == 'geom':
            return 2.0 ** ((math.log2(1.0) - math.log2(self.start_rate)) / self.grow_epochs * self.epoch + math.log2(self.start_rate))
        elif self.grow_fn[:5] == 'root-' and self.grow_fn[5:].isnumeric():
            p = int(self.grow_fn[5:])
            return (self.start_rate ** p + (1.0 - self.start_rate ** p) / self.grow_epochs * self.epoch) ** 0.5
        else:
            raise NotImplementedError()


    def _loss_measure(self):
        return torch.cat([self.criterion(self.net(
            data[0].to(self.device)), data[1].to(self.device)).detach() 
            for data in DataLoader(self.dataset, self.batch_size)])


    def _loss_reweight(self, loss, threshold):
        mask = loss < threshold
        if self.weight_fn == 'linear':
            return mask * (1.0 - loss / threshold)
        elif self.weight_fn == 'logarithmic':
            return mask * (torch.log(loss + 1.0 - threshold) / torch.log(1.0 - threshold))
        elif self.weight_fn == 'logistic':
            return (1.0 + torch.exp(-threshold)) / (1.0 + torch.exp(loss - threshold))
        elif self.weight_fn[:11] == 'polynomial-' and self.weight_fn[11:].isnumeric():
            t = int(self.weight_fn[11:])
            return mask * ((1.0 - loss / threshold) ** 1.0 / (t - 1.0))      
        else:
            raise NotImplementedError()


class SelfPacedTrainer(BaseTrainer):
    def __init__(self, data_name, net_name, device_name, random_seed, 
                 start_rate, grow_epochs, grow_fn, weight_fn):
        
        if data_name in ['cifar10']:
            cl = SelfPaced(
                start_rate, grow_epochs, grow_fn, weight_fn,
                torch.nn.CrossEntropyLoss(reduction='none'),
                torch.device(device_name),
            )
        else:
            raise NotImplementedError()

        super(SelfPacedTrainer, self).__init__(
            data_name, net_name, device_name, random_seed, cl
        )
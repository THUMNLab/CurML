import math
from torch.utils.data import Subset, DataLoader

from .base import BaseTrainer, BaseCL



class LambdaStep(BaseCL):
    def __init__(self, start_rate, grow_epochs, grow_fn):
        super(LambdaStep, self).__init__()

        self.name = 'lambdastep'
        self.start_rate = start_rate
        self.grow_epochs = grow_epochs
        self.grow_fn = grow_fn


    def data_curriculum(self, loader):
        super().data_curriculum(loader)

        data_rate = min(1.0, self._subset_grow())
        data_size = int(self.data_size * data_rate)

        dataset = Subset(self.dataset, tuple(range(data_size)))
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


class LambdaStepTrainer(BaseTrainer):
    def __init__(self, data_name, net_name, 
                 device_name, random_seed, 
                 start_rate, grow_epochs, grow_fn):
        
        cl = LambdaStep(start_rate, grow_epochs, grow_fn)
        
        super(LambdaStepTrainer, self).__init__(
            data_name, net_name, device_name, random_seed, cl
        )
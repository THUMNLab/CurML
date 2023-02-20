import math
import random
from torch.utils.data import Subset, DataLoader

from .base import BaseTrainer, BaseCL



class LambdaStep(BaseCL):
    """Lambda Step CL Algorithm. A predefined CL with a continuous scheduler.

    Source:
        On the power of curriculum learning in training deep networks. http://proceedings.mlr.press/v97/hacohen19a/hacohen19a.pdf

    Attributes:
        name, dataset, data_size, batch_size, n_batches: Base class attributes.
        epoch: An integer count of current training epoch.
        start_rate: An float indicating the initial proportion of the sampled data instances.
        grow_epochs: An integer for the epoch when the proportion of sampled data reaches 1.0.
        grow_fn: Pacing function or Competence function that how the proportion of sampled data grows.
        not_sorted: An boolean of whether the data has been sorted by difficulty. Default: False.
    """
    def __init__(self, start_rate, grow_epochs, grow_fn, not_sorted=True):
        super(LambdaStep, self).__init__()

        self.name = 'lambda_step'
        self.epoch = 0
        
        self.start_rate = start_rate
        self.grow_epochs = grow_epochs
        self.grow_fn = grow_fn
        self.not_sorted = not_sorted


    def data_prepare(self, loader):
        super().data_prepare(loader)

        self.data_indices = list(range(self.data_size))         # Assume the data is sorted by difficulty.
        if self.not_sorted:
            random.shuffle(self.data_indices)                   # Else shuffle data to simulate data sorting by difficulty.


    def data_curriculum(self, loader):
        """Measure difficulty and schedule training.
        
        Measure difficulty: Assume the data is sorted by difficulty.
        Schedule training: Add more difficult data to the training set every epoch.
        """
        self.epoch += 1
        
        data_rate = min(1.0, self._subset_grow())               # Current proportion of sampled data.
        data_size = int(self.data_size * data_rate)             # Current number of sampled data.
        data_indices = self.data_indices[:data_size]            # Current indices of samples data.

        dataset = Subset(self.dataset, data_indices)
        return DataLoader(dataset, self.batch_size, shuffle=True)


    def _subset_grow(self):
        if self.grow_fn == 'linear':                            # Linear Function.
            return self.start_rate + (1.0 - self.start_rate) / self.grow_epochs * self.epoch
        elif self.grow_fn == 'geom':                            # Geometric Function.
            return 2.0 ** ((math.log2(1.0) - math.log2(self.start_rate)) / self.grow_epochs * self.epoch + math.log2(self.start_rate))
        elif self.grow_fn[:5] == 'root-' and self.grow_fn[5:].isnumeric():
            p = int(self.grow_fn[5:])                           # Root-p Function.
            return (self.start_rate ** p + (1.0 - self.start_rate ** p) / self.grow_epochs * self.epoch) ** 0.5
        else:
            raise NotImplementedError()


class LambdaStepTrainer(BaseTrainer):
    def __init__(self, data_name, net_name, device_name, num_epochs, random_seed, 
                 start_rate, grow_epochs, grow_fn, not_sorted=False):
        
        cl = LambdaStep(start_rate, grow_epochs, grow_fn, not_sorted)
        
        super(LambdaStepTrainer, self).__init__(
            data_name, net_name, device_name, num_epochs, random_seed, cl)
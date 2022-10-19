import random
from torch.utils.data import Subset, DataLoader

from .base import BaseTrainer, BaseCL



class BabyStep(BaseCL):
    """Baby Step Algorithm. A predefined CL with a discrete scheduler.

    Sources:
        Curriculum learning. https://dl.acm.org/doi/pdf/10.1145/1553374.1553380
        Visualizing and understanding curriculum learning for long short-term memory networks. https://arxiv.org/pdf/1611.06204.pdf
        From baby steps to leapfrog: How "less is more" in unsupervised dependency parsing. https://aclanthology.org/N10-1116.pdf

    Attributes:
        name, dataset, data_size, batch_size, n_batches: Base class attributes.
        epoch: An integer count of current training epoch.
        start_rate: An float for the initial proportion of the sampled data instances.
        grow_rate: An float for the growth proportion of the sampled data instance.
        grow_interval: An integer for the number of training set growth interval.
        not_sorted: An boolean of whether the data has been sorted by difficulty. Default: False.
    """
    def __init__(self, start_rate, grow_rate, grow_interval, not_sorted=False):
        super(BabyStep, self).__init__()

        self.name = 'baby_step'
        self.epoch = 0

        self.start_rate = start_rate
        self.grow_rate = grow_rate
        self.grow_interval = grow_interval
        self.not_sorted = not_sorted

    
    def data_prepare(self, loader):
        super().data_prepare(loader)

        self.data_indices = list(range(self.data_size))         # Assume the data is sorted by difficulty.
        if self.not_sorted:
            random.shuffle(self.data_indices)                   # Else shuffle data to simulate data sorting by difficulty.


    def data_curriculum(self, loader):
        """Measure difficulty and schedule training.
        
        Measure difficulty: Assume the data is sorted by difficulty.
        Schedule training: Add more difficult data to the training set every growth interval.
        """
        self.epoch += 1

        data_rate = min(1.0, self._subset_grow())               # Current proportion of sampled data.
        data_size = int(self.data_size * data_rate)             # Current number of sampled data.
        data_indices = self.data_indices[:data_size]            # Current indices of samples data.

        dataset = Subset(self.dataset, data_indices)
        return DataLoader(dataset, self.batch_size, shuffle=True)


    def _subset_grow(self):
        """Every growth interval add growth rate data"""
        return self.start_rate + self.grow_rate * ((self.epoch - 1) // self.grow_interval + 1)


class BabyStepTrainer(BaseTrainer):
    def __init__(self, data_name, net_name, device_name, num_epochs, random_seed, 
                 start_rate, grow_rate, grow_interval, not_sorted=False):
        
        cl = BabyStep(start_rate, grow_rate, grow_interval, not_sorted)
        
        super(BabyStepTrainer, self).__init__(
            data_name, net_name, device_name, num_epochs, random_seed, cl)
from torch.utils.data import Subset, DataLoader

from .base import BaseTrainer, BaseCL



class BabyStep(BaseCL):
    def __init__(self, start_rate, grow_rate, grow_interval):
        super(BabyStep, self).__init__()

        self.name = 'babystep'
        self.epoch = 0

        self.start_rate = start_rate
        self.grow_rate = grow_rate
        self.grow_interval = grow_interval


    def data_curriculum(self, loader):
        self.epoch += 1

        data_rate = min(1.0, self._subset_grow())
        data_size = int(self.data_size * data_rate)

        dataset = Subset(self.dataset, tuple(range(data_size)))
        return DataLoader(dataset, self.batch_size, shuffle=True)


    def _subset_grow(self):
        return self.start_rate + self.grow_rate * ((self.epoch - 1) // self.grow_interval + 1)


class BabyStepTrainer(BaseTrainer):
    def __init__(self, data_name, net_name, device_name, random_seed, 
                 start_rate, grow_rate, grow_interval):
        
        cl = BabyStep(start_rate, grow_rate, grow_interval)
        
        super(BabyStepTrainer, self).__init__(
            data_name, net_name, device_name, random_seed, cl
        )
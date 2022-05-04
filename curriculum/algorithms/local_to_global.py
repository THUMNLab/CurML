from .base import BaseTrainer, BaseCL



class LocalToGlobal(BaseCL):
    def __init__(self, class_size, start_size, 
                 grow_size, grow_interval):
        super(LocalToGlobal, self).__init__()

        self.name = 'localtoglobal'
        self.epoch = 0

        self.class_size = class_size
        self.start_size = start_size
        self.grow_size = grow_size
        self.grow_interval = grow_interval
        

    def model_prepare(self, net, device, epochs, 
                      criterion, optimizer, lr_scheduler):
        self.lr_scheduler = lr_scheduler
        self.init_optimizer = self.lr_scheduler.state_dict()


    def data_curriculum(self, loader):
        self.epoch += 1

        class_size = min(self.class_size, self._subclass_grow())
        return super().data_curriculum()

    
    def _subclass_grow(self):
        return self.start_size + self.grow_size * ((self.epoch - 1) // self.grow_interval + 1)


class LocalToGlobalTrainer(BaseTrainer):
    def __init__(self, data_name, net_name, device_name, random_seed,
                 start_size, grow_size, grow_interval):
        
        class_size_dict = {'cifar10': 10}
        cl = LocalToGlobal(class_size_dict[data_name], 
                           start_size, grow_size, grow_interval)

        super(LocalToGlobalTrainer, self).__init__(
            data_name, net_name, device_name, random_seed, cl
        )
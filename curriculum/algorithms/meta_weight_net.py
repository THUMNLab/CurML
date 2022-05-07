from .base import BaseTrainer, BaseCL



class MetaWeightNet(BaseCL):
    def __init__(self, ):
        super(MetaWeightNet, self).__init__()

        self.name = 'metaweightnet'
        



class MetaWeightNetTrainer(BaseTrainer):
    def __init__(self, data_name, net_name, device_name, num_epochs, random_seed):
        
        cl = MetaWeightNet()

        super(MetaWeightNetTrainer, self).__init__(
            data_name, net_name, device_name, num_epochs, random_seed, cl
        )
from .base import BaseTrainer, BaseCL



class MetaReweight(BaseCL):
    def __init__(self, ):
        super(MetaReweight, self).__init__()

        self.name = 'metareweight'
        



class MetaReweightTrainer(BaseTrainer):
    def __init__(self, data_name, net_name, device_name, num_epochs, random_seed):
        
        cl = MetaReweight()

        super(MetaReweightTrainer, self).__init__(
            data_name, net_name, device_name, num_epochs, random_seed, cl
        )
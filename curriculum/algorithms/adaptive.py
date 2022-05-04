from .base import BaseTrainer, BaseCL



class Adaptive(BaseCL):
    def __init__(self, ):
        super(Adaptive, self).__init__()

        self.name = 'adaptive'
        



class AdaptiveTrainer(BaseTrainer):
    def __init__(self, data_name, net_name, device_name, random_seed):
        
        cl = Adaptive()

        super(AdaptiveTrainer, self).__init__(
            data_name, net_name, device_name, random_seed, cl
        )
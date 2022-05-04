from .base import BaseTrainer, BaseCL



class LocalToGlobal(BaseCL):
    def __init__(self, ):
        super(LocalToGlobal, self).__init__()

        self.name = 'localtoglobal'
        



class LocalToGlobalTrainer(BaseTrainer):
    def __init__(self, data_name, net_name, device_name, random_seed):
        
        cl = LocalToGlobal()

        super(LocalToGlobalTrainer, self).__init__(
            data_name, net_name, device_name, random_seed, cl
        )
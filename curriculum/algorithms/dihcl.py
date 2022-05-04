from .base import BaseTrainer, BaseCL



class DIHCL(BaseCL):
    def __init__(self, ):
        super(DIHCL, self).__init__()

        self.name = 'dihcl'
        



class DIHCLTrainer(BaseTrainer):
    def __init__(self, data_name, net_name, device_name, random_seed):
        
        cl = DIHCL()

        super(DIHCLTrainer, self).__init__(
            data_name, net_name, device_name, random_seed, cl
        )
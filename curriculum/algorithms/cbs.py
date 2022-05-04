from .base import BaseTrainer, BaseCL



class CBS(BaseCL):
    def __init__(self, ):
        super(CBS, self).__init__()

        self.name = 'cbs'
        



class CBSTrainer(BaseTrainer):
    def __init__(self, data_name, net_name, device_name, random_seed):
        
        cl = CBS()

        super(CBSTrainer, self).__init__(
            data_name, net_name, device_name, random_seed, cl
        )
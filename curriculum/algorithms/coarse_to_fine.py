from .base import BaseTrainer, BaseCL



class CoarseToFine(BaseCL):
    def __init__(self, ):
        super(CoarseToFine, self).__init__()

        self.name = 'coarsetofine'
        



class CoarseToFineTrainer(BaseTrainer):
    def __init__(self, data_name, net_name, device_name, random_seed):
        
        cl = CoarseToFine()

        super(CoarseToFineTrainer, self).__init__(
            data_name, net_name, device_name, random_seed, cl
        )
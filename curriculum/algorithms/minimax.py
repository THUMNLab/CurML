from .base import BaseTrainer, BaseCL



class Minimax(BaseCL):
    def __init__(self, ):
        super(Minimax, self).__init__()

        self.name = 'minimax'
        



class MinimaxTrainer(BaseTrainer):
    def __init__(self, data_name, net_name, device_name, random_seed):
        
        cl = Minimax()

        super(MinimaxTrainer, self).__init__(
            data_name, net_name, device_name, random_seed, cl
        )
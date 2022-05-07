from .base import BaseTrainer, BaseCL



class ScreenerNet(BaseCL):
    def __init__(self, ):
        super(ScreenerNet, self).__init__()

        self.name = 'screenernet'
        



class ScreenerNetTrainer(BaseTrainer):
    def __init__(self, data_name, net_name, device_name, num_epochs, random_seed):
        
        cl = ScreenerNet()

        super(ScreenerNetTrainer, self).__init__(
            data_name, net_name, device_name, num_epochs, random_seed, cl
        )
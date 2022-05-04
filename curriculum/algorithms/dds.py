from .base import BaseTrainer, BaseCL



class DDS(BaseCL):
    def __init__(self, ):
        super(DDS, self).__init__()

        self.name = 'dds'
        



class DDSTrainer(BaseTrainer):
    def __init__(self, data_name, net_name, device_name, random_seed):
        
        cl = DDS()

        super(DDSTrainer, self).__init__(
            data_name, net_name, device_name, random_seed, cl
        )
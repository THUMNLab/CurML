from .base import BaseTrainer, BaseCL



class RLTeacher(BaseCL):
    def __init__(self, ):
        super(RLTeacher, self).__init__()

        self.name = 'rlteacher'
        



class RLTeacherTrainer(BaseTrainer):
    def __init__(self, data_name, net_name, device_name, num_epochs, random_seed):
        
        cl = RLTeacher()

        super(RLTeacherTrainer, self).__init__(
            data_name, net_name, device_name, num_epochs, random_seed, cl
        )
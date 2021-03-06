from .base import BaseTrainer
from .self_paced import SelfPaced



class TransferTeacher(SelfPaced):
    def __init__(self, start_rate, grow_epochs, 
                 grow_fn, weight_fn, teacher_net):
        super(TransferTeacher, self).__init__(
            start_rate, grow_epochs, grow_fn, weight_fn)

        self.name = 'transferteacher'
        self.net = teacher_net
        self.data_loss = None


    def _loss_measure(self):
        if self.data_loss is None:
            self.data_loss = super()._loss_measure()
        return self.data_loss



class TransferTeacherTrainer(BaseTrainer):
    def __init__(self, data_name, net_name, device_name, num_epochs, random_seed, 
                 start_rate, grow_epochs, grow_fn, weight_fn, teacher_net):

        cl = TransferTeacher(
            start_rate, grow_epochs, grow_fn, weight_fn, teacher_net
        )

        super(TransferTeacherTrainer, self).__init__(
            data_name, net_name, device_name, num_epochs, random_seed, cl
        )
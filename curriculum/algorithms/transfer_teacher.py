from .base import BaseTrainer
from .self_paced import SelfPaced



class TransferTeacher(SelfPaced):
    """Transfer Teacher CL Algorithm.

    It is inherited from the Self-Paced Learning, but the data difficulty is decided by a pre-trained teacher. 
    Curriculum learning by transfer learning: Theory and experiments with deep networks. http://proceedings.mlr.press/v80/weinshall18a/weinshall18a.pdf
    
    Attributes:
        name, dataset, data_size, batch_size, n_batches: Base class attributes.
        epoch, start_rate, grow_epochs, grow_fn, device, criterion, weights: SelfPaced class attributes.
        net: A pre-trained teacher net.
        data_loss: save the loss calculated by the teacher net.
    """
    def __init__(self, start_rate, grow_epochs, 
                 grow_fn, weight_fn, teacher_net):
        super(TransferTeacher, self).__init__(
            start_rate, grow_epochs, grow_fn, weight_fn)

        self.name = 'transfer_teacher'
        self.net = teacher_net
        self.data_loss = None


    def _loss_measure(self):
        """Only calculate the data loss once, because the teacher net is fixed and the loss will not be changed."""
        if self.data_loss is None:
            self.data_loss = super()._loss_measure()
        return self.data_loss



class TransferTeacherTrainer(BaseTrainer):
    def __init__(self, data_name, net_name, device_name, num_epochs, random_seed, 
                 start_rate, grow_epochs, grow_fn, weight_fn, teacher_net):

        cl = TransferTeacher(start_rate, grow_epochs, grow_fn, weight_fn, teacher_net)

        super(TransferTeacherTrainer, self).__init__(
            data_name, net_name, device_name, num_epochs, random_seed, cl)
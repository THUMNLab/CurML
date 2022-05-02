import torch

from .self_paced import SelfPacedTrainer, SelfPaced



class TransferTeacher(SelfPaced):
    def __init__(self, start_rate, grow_epochs, grow_fn,  
                 weight_fn, teacher_net, criterion, device):
        super(TransferTeacher, self).__init__(
            start_rate, grow_epochs, grow_fn, 
            weight_fn, criterion, device)

        self.name = 'transferteacher'
        self.data_loss = None
        self.net = teacher_net
        self.net.to(device)


    def model_curriculum(self, net):
        return net


    def _loss_measure(self):
        if self.data_loss is None:
            self.data_loss = super()._loss_measure()
        return self.data_loss



class TransferTeacherTrainer(SelfPacedTrainer):
    def __init__(self, data_name, net_name, device_name, random_seed, 
                 start_rate, grow_epochs, weight_fn, grow_fn, teacher_net):
        
        if data_name in ['cifar10']:
            cl = TransferTeacher(
                start_rate, grow_epochs, grow_fn, weight_fn, teacher_net,
                torch.nn.CrossEntropyLoss(reduction='none'),
                torch.device(device_name),
            )
        else:
            raise NotImplementedError()

        super(TransferTeacherTrainer, self).__init__(
            data_name, net_name, device_name, random_seed, cl
        )
import torch

from .base import BaseTrainer, BaseCL
from .utils import SparseSGD


class DataParameters(BaseCL):
    def __init__(self, class_size, device):
        super(DataParameters, self).__init__()

        self.name = 'dataparameters'
        self.data_weights = None
        self.data_optimizer = None
        self.class_weights = None
        self.class_optimizer = None


    def data_curriculum(self, loader):
        super().data_curriculum(loader)

        if self.data_optimizer is None:
            self.data_weights = torch.ones(self.data_size, requires_grad=True
            
            
    class_parameters = torch.tensor(np.ones(num_class) * np.log(1.0),
                                    dtype=torch.float32,
                                    requires_grad=True,
                                    device=device)
    optimizer_class_param = SparseSGD([class_parameters],
                                      lr=0.1,
                                      momentum=0.9,
                                      skip_update_zero_grad=True)

    # instance-parameter
    inst_parameters = torch.tensor(np.ones(num_instance) * np.log(1.0),
                                   dtype=torch.float32,
                                   requires_grad=True,
                                   device=device)
    optimizer_inst_param = SparseSGD([inst_parameters],
                                     lr=0.2,
                                     momentum=0.9,
                                     skip_update_zero_grad=True)


    def loss_curriculum(self, outputs, labels, criterion, indices):
        return
        


class DataParametersTrainer(BaseTrainer):
    def __init__(self, data_name, net_name, 
                 device_name, random_seed):
        
        cl = DataParameters()
        
        super(DataParametersTrainer, self).__init__(
            data_name, net_name, device_name, random_seed, cl
        )





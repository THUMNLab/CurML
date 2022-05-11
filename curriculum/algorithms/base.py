import torch
from torch.utils.data import Dataset, DataLoader

from ..solvers import *



class BaseCL():

    class CLDataset(Dataset):
        def __init__(self, dataset):
            self.dataset = dataset

        def __getitem__(self, index):
            data = self.dataset[index]
            return [part for part in data] + [index]

        def __len__(self):
            return len(self.dataset)


    def __init__(self):
        self.name = 'base'


    def data_prepare(self, loader):
        self.dataset = self.CLDataset(loader.dataset)
        self.data_size = len(self.dataset)
        self.batch_size = loader.batch_size
        self.n_batches = (self.data_size - 1) // self.batch_size + 1


    def model_prepare(self, net, device, epochs, 
                      criterion, optimizer, lr_scheduler):
        pass


    def data_curriculum(self, loader):
        return DataLoader(self.dataset, self.batch_size, shuffle=True)


    def model_curriculum(self, net):
        return net


    def loss_curriculum(self, criterion, outputs, labels, indices):
        return torch.mean(criterion(outputs, labels))



class BaseTrainer():

    def __init__(self, data_name, net_name, device_name, 
                 num_epochs, random_seed, cl=BaseCL()):
        
        if data_name in ['cifar10']:
            self.trainer = ImageClassifier(
                data_name, net_name, device_name, num_epochs, random_seed,
                cl.name, cl.data_prepare, cl.model_prepare,
                cl.data_curriculum, cl.model_curriculum, cl.loss_curriculum,
            )
        else:
            raise NotImplementedError()
        

    def fit(self):
        return self.trainer.fit()


    def evaluate(self, net_dir=None):
        return self.trainer.evaluate(net_dir)

    
    def export(self, net_dir=None):
        return self.trainer.export(net_dir)
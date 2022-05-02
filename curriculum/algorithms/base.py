import torch
from torch.utils.data import Dataset, DataLoader

from ..solvers import *



class BaseCL():

    class CLDataset(Dataset):
        def __init__(self, dataset, weights=None):
            self.dataset = dataset
            self.weights = torch.ones(len(dataset)) \
                if weights is None else weights

        def __getitem__(self, index):
            data = self.dataset[index]
            weights = self.weights[index]
            return [part for part in data] + [weights]

        def __len__(self):
            return len(self.dataset)


    def __init__(self):
        self.name = 'base'
        self.epoch = 0


    def data_curriculum(self, loader):
        self.epoch += 1

        self.dataset = self.CLDataset(loader.dataset)
        self.data_size = len(self.dataset)
        self.batch_size = loader.batch_size
        self.n_batches = (self.data_size - 1) // self.batch_size + 1

        return DataLoader(self.dataset, self.batch_size, shuffle=True)


    def model_curriculum(self, net):
        return net


    def loss_curriculum(self, outputs, labels, criterion, weights):
        return torch.mean(criterion(outputs, labels) * weights)



class BaseTrainer():

    def __init__(self, data_name, net_name, 
                 device_name, random_seed, 
                 cl=BaseCL()):
        
        if data_name in ['cifar10']:
            self.trainer = ImageClassifier(
                data_name, net_name, 
                device_name, random_seed,
                cl.name, cl.data_curriculum, 
                cl.model_curriculum, cl.loss_curriculum,
            )
        

    def fit(self):
        self.trainer.fit()


    def evaluate(self):
        self.trainer.evaluate()
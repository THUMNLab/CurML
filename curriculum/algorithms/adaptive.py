import math
import torch
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from .base import BaseTrainer, BaseCL



class Adaptive(BaseCL):
    """
    
    Adaptive Curriculum Learning. https://openaccess.thecvf.com/content/ICCV2021/papers/Kong_Adaptive_Curriculum_Learning_ICCV_2021_paper.pdf
    """
    def __init__(self, num_classes, pace_p, pace_q, pace_r, inv,
                 alpha, gamma, gamma_decay, bottom_gamma, pretrained_net):
        super(Adaptive, self).__init__()

        self.name = 'adaptive'

        self.epoch = 0
        self.pace_p = pace_p
        self.epoch_size = pace_p
        self.pace_q = pace_q
        self.pace_r = pace_r
        self.cnt = 0
        self.inv = inv
        self.alpha = alpha
        self.gamma = gamma
        self.gamma_decay = gamma_decay
        self.bottom_gamma = bottom_gamma
        self.num_classes = num_classes
        self.pretrained_model = pretrained_net


    def data_prepare(self, loader):
        self.dataloader = loader
        self.dataset = self.CLDataset(loader.dataset)
        self.data_size = len(self.dataset)
        self.batch_size = loader.batch_size
        self.n_batches = (self.data_size - 1) // self.batch_size + 1


    def model_prepare(self, net, device, epochs, criterion, optimizer, lr_scheduler):
        self.device = device
        self.model = net
        self.critertion = criterion
        self.total_epoch = epochs
    

    def data_curriculum(self, loader):
        if self.epoch == 0:
            self.pretrained_model.to(self.device)
            self.difficulty = torch.Tensor().to(self.device)
            self.pretrained_output = torch.Tensor().to(self.device)
            self.data_indice = torch.arange(self.data_size)
            self.crossEntrophy = torch.nn.CrossEntropyLoss(reduction='none')
            self.KLloss = torch.nn.KLDivLoss(reduction='batchmean')
            self._set_initial_difficulty()
            self.pretrained_difficulty = self.difficulty

        self.epoch += 1
        self.cnt = 0

        self.epoch_size = self.data_size * min(self.pace_p * (self.pace_q ** int(math.floor(self.epoch / self.pace_r))), 1)
        self.epoch_size = int(self.epoch_size)
        data_sort = torch.argsort(self.difficulty)
        self.data_indice = data_sort[0 : self.epoch_size]
        dataset = Subset(self.dataset, self.data_indice)
        dataloader = DataLoader(dataset, self.batch_size, shuffle=False)

        if self.epoch % self.inv == 0:
            self._difficulty_measurer()

            # gradually reduce gamma which is the balancing parameter controling how much the knowledge learned from the pretrained model
            if self.gamma_decay is not None:
                self.gamma = max(self.bottom_gamma, self.gamma - self.gamma_decay)

        return dataloader
        
    
    def loss_curriculum(self, criterion, outputs, labels, indices):
        losses = torch.mean(criterion(outputs, labels))
        epoch_pretrained_output = torch.Tensor().to(self.device)
        for indice in self.data_indice[self.cnt : (self.cnt + self.batch_size)]:
            epoch_pretrained_output = torch.cat((epoch_pretrained_output, self.pretrained_output[int(indice)]), 0)
        epoch_pretrained_output = epoch_pretrained_output.view(-1, 10)
        epoch_pretrained_output = F.softmax(epoch_pretrained_output, dim=1)

        output = F.softmax(outputs, dim=1)
        kl_divergence = self.KLloss(output, epoch_pretrained_output)
        losses = losses + self.gamma * kl_divergence
        self.cnt += self.batch_size
        return losses      


    def _set_initial_difficulty(self):
    
        self.pretrained_model.eval()
        for step, data in enumerate(self.dataloader):
            inputs = data[0].to(self.device)
            labels = data[1].to(self.device)
            with torch.no_grad():
                outputs = self.pretrained_model(inputs)
            self.pretrained_output = torch.cat((self.pretrained_output, outputs), 0)
            loss = self.crossEntrophy(outputs, labels)
            self.difficulty = torch.cat((self.difficulty, loss), 0)


    def _difficulty_measurer(self):
    
        current_difficulty = torch.Tensor().to(self.device)

        for step, data in enumerate(self.dataloader):
            with torch.no_grad():
                outputs = self.model(data[0].to(self.device))
            loss = self.crossEntrophy(outputs, data[1].to(self.device)).detach()
            current_difficulty = torch.cat((current_difficulty, loss), 0)
        
        self.difficulty = (1 - self.alpha) * self.difficulty + self.alpha * current_difficulty


class AdaptiveTrainer(BaseTrainer):
    def __init__(self, data_name, net_name, device_name, num_epochs, random_seed,
                 num_classes, pace_p, pace_q, pace_r, inv,
                 alpha, gamma, gamma_decay, bottom_gamma, pretrained_net):
        
        cl = Adaptive(num_classes, pace_p, pace_q, pace_r, inv,
                 alpha, gamma, gamma_decay, bottom_gamma, pretrained_net)

        super(AdaptiveTrainer, self).__init__(
            data_name, net_name, device_name, num_epochs, random_seed, cl)


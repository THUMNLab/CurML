import numpy as np
from scipy.special import lambertw
import torch

from .base import BaseTrainer, BaseCL



class Superloss(BaseCL):
    """
    
    Superloss: A generic loss for robust curriculum learning. https://proceedings.neurips.cc/paper/2020/file/2cfa8f9e50e0f510ede9d12338a5f564-Paper.pdf
    """
    def __init__(self, tau, lam, fac):
        super(Superloss, self).__init__()

        self.name = 'superloss'
        self.tau = tau
        self.lam = lam
        self.fac = fac


    def model_prepare(self, net, device, epochs, 
                      criterion, optimizer, lr_scheduler):
        self.device = device


    def loss_curriculum(self, criterion, outputs, labels, indices):
        loss = criterion(outputs, labels)
        origin_loss = loss.detach().cpu().numpy()

        if self.fac > 0.0:
            self.tau = self.fac * origin_loss.mean() + (1.0 - self.fac) * self.tau

        beta = (origin_loss - self.tau) / self.lam
        gamma = -2.0 / np.exp(1.0)
        sigma = np.exp(-lambertw(0.5 * np.maximum(beta, gamma))).real
        sigma = torch.from_numpy(sigma).to(self.device)
        super_loss = (loss - self.tau) * sigma + self.lam * (torch.log(sigma) ** 2)
        return torch.mean(super_loss)



class SuperlossTrainer(BaseTrainer):
    def __init__(self, data_name, net_name, device_name, 
                 num_epochs, random_seed, tau, lam, fac):
        
        cl = Superloss(tau, lam, fac)
        
        super(SuperlossTrainer, self).__init__(
            data_name, net_name, device_name, num_epochs, random_seed, cl)
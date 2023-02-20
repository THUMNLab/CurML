import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .base import BaseTrainer, BaseCL



class DIHCL(BaseCL):
    """
    
    Curriculum learning by dynamic instance hardness. https://proceedings.neurips.cc/paper/2020/file/62000dee5a05a6a71de3a6127a68778a-Paper.pdf
    """
    def __init__(self, warm_epoch, discount_factor, decay_rate, 
                 bottom_size, type, sample_type, cei):
        super(DIHCL, self).__init__()

        self.name = 'dihcl'
        self.discount_factor = discount_factor
        self.bottom_size = bottom_size
        self.decay_rate = decay_rate
        self.epoch = 0
        self.rate = 1
        self.type = type
        self.sample_type = sample_type
        self.lr_array = np.array([0, ])
        
        self.epoch_lr = 0
        self.warm_epoch = warm_epoch
        assert self.warm_epoch >= 1, \
            'Assert Error: there should be at least 1 warm epoch.'

        if self.sample_type == 'beta':
            self.cei = cei


    def data_prepare(self, loader):
        super().data_prepare(loader)

        self.dih_loss = np.zeros(self.data_size)
        self.train_set = np.arange(self.data_size)
        self.probability = np.ones(self.data_size) / self.data_size
        self.old_loss = np.zeros(self.data_size)
        self.data_index = np.zeros(self.data_size)


    def model_prepare(self, net, device, epochs, 
                      criterion, optimizer, lr_scheduler):
        self.lr_scheduler = lr_scheduler
    

    def data_curriculum(self, loader):
        self.cnt = 0
        self.epoch += 1
        if self.epoch > 1:
            self.lr_array = np.append(self.lr_array, 
                self.lr_array[len(self.lr_array) - 1] + self.epoch_lr
            ) # prefix sum

        if self.epoch <= self.warm_epoch:
            self.train_set = np.arange(self.data_size)
        else:
            self._probability_regularize()
            self.rate = max(self.rate * self.decay_rate, self.bottom_size)
            select = int(np.floor(self.rate * self.data_size))
            self.train_set = np.random.choice(
                self.data_size, select, p=self.probability, replace=False
            )

        dataset = Subset(self.dataset, self.train_set)
        return DataLoader(dataset, self.batch_size, shuffle=False)


    def _probability_regularize(self):
        if self.sample_type == 'rand':
            self.probability = self.dih_loss

        elif self.sample_type == 'exp':
            self.probability = np.exp(np.sqrt(2 * np.log(self.data_size) / self.data_size) * self.dih_loss)

        elif self.sample_type == 'beta':
            temp = (self.dih_loss == 0) # remove 0 value sample, since it is unlikely to be fetched
            self.dih_loss[temp] = 0.01
            if self.cei < np.max(self.dih_loss):
                self.cei = np.max(self.dih_loss) + 0.1
            ceiling = self.cei * np.ones(self.data_size)
            self.probability = np.random.beta(self.dih_loss, ceiling - self.dih_loss)
            self.probability[temp] = 0
        else:
            raise NotImplementedError()

        self.probability[(self.probability == 0)] = 1e-6 # change zero probability to a small value
        self.probability /= np.sum(self.probability)


    def loss_curriculum(self, criterion, outputs, labels, indices):
        losses = criterion(outputs, labels)
        self.epoch_lr = self.lr_scheduler.get_last_lr()[0]
        if self.type == 'prediction_flip':
            cnt = 0
            assert(len(losses) == len(labels))

        for loss in losses:
            # Normalize losses according to the learning rate 
            # then compute the DIH losses
            
            index = self.train_set[self.cnt]
            if self.type == 'loss':
                lr_sum = self.epoch_lr
                process_loss = loss.cpu().detach().numpy()
                lr_sum = max(lr_sum, 1e-3)
                process_loss = process_loss / lr_sum

            elif self.type == 'loss_change':
                if int(self.data_index[index]) == 0:
                    lr_sum = self.lr_array[len(self.lr_array) - 1] + self.epoch_lr
                else:
                    lr_sum = self.lr_array[len(self.lr_array) - 1] - self.lr_array[int(self.data_index[index]) - 1] + self.epoch_lr
                process_loss = abs(loss.cpu().detach().numpy() - self.old_loss[index])
                lr_sum = max(lr_sum, 1e-3)
                process_loss = process_loss / lr_sum
                self.old_loss[index] = loss.cpu().detach().numpy()
                self.data_index[index] = self.epoch

            elif self.type == 'prediction_flip':
                _, prediction_label = torch.max(loss, dim=0)
                if int(self.data_index[index]) == 0:
                    lr_sum = self.lr_array[len(self.lr_array) - 1] + self.epoch_lr
                else:
                    lr_sum = self.lr_array[len(self.lr_array) - 1] - self.lr_array[int(self.data_index[index]) - 1] + self.epoch_lr
                process_loss = abs((prediction_label.item() == labels[cnt].item()) - self.old_loss[index])
                lr_sum = max(lr_sum, 1e-3)
                process_loss = process_loss / lr_sum
                self.old_loss[index] = abs(prediction_label.item() == labels[cnt].item())
                self.data_index[index] = self.epoch
                cnt += 1
            else:
                return NotImplementedError()

            self.dih_loss[index] = (1 - self.discount_factor) * self.dih_loss[index] \
                                    + self.discount_factor * process_loss
            assert(self.dih_loss[index] >= 0)
            self.cnt += 1

        return torch.mean(losses)




class DIHCLTrainer(BaseTrainer):
    def __init__(self, data_name, net_name, device_name, num_epochs, random_seed,
                 warm_epoch, discount_factor, decay_rate, bottom_size,
                 type, sample_type, cei):
        
        cl = DIHCL(warm_epoch, discount_factor, decay_rate, bottom_size,
                 type, sample_type, cei)

        super(DIHCLTrainer, self).__init__(
            data_name, net_name, device_name, num_epochs, random_seed, cl)

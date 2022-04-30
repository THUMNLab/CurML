import os
import time

import random
import numpy as np

import torch
import torchvision


from ..datasets import get_dataset
from ..backbones import get_net
from ..utils import get_logger



class ImageClassifier:
    def __init__(self, algorithm_name, data_name, 
                 net_name, device_name, random_seed):

        self._init_dataloader(data_name)
        self._init_model(net_name, device_name)
        self._init_logger(algorithm_name, data_name, 
                          net_name, random_seed)


    def _init_dataloader(self, data_name):
        train_dataset, valid_dataset, test_dataset = \
            get_dataset('./data', data_name)

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=128, shuffle=True,
            num_workers=2, pin_memory=True,
        )
        self.valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=100, shuffle=False,
            num_workers=2, pin_memory=True,
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=100, shuffle=False,
            num_workers=2, pin_memory=True,
        )


    def _init_model(self, net_name, device_name):
        self.net = get_net(net_name)
        self.device = torch.device(device_name \
            if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

        self.epochs = 200
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4
        )
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, Tmax=self.epochs, eta_min=1e-6
        )

    
    def _init_logger(self, algorithm_name, data_name, 
                     net_name, random_seed):
        self.log_interval = 10

        log_info = '%s-%s-%s-%d' % (
            algorithm_name, data_name, net_name, random_seed
        )
        self.log_dir = os.path.join('./runs', log_info)
        if not os.path.exists('./runs'): os.mkdir('./runs')
        assert not os.path.exists(self.log_dir), \
            'Assert Error: the directory %s has already existed' % (self.log_dir)
        os.mkdir(self.log_dir)
        log_file = os.path.join(self.log_dir, 'train.log')

        self.logger = get_logger(log_file)


    def _train_one_epoch(self, epoch):
        t = time.time()
        total = 0
        running_loss = 0.0

        self.net.train()
        # for step, data in enumerate(curriculum.data_scheduler()): # curriculum part
        #     inputs = data[0].to(device)
        #     labels = data[1].to(device)

        #     optimizer.zero_grad()
        #     outputs = net(inputs)
        #     loss = criterion(outputs, labels)
        #     loss.backward()
        #     optimizer.step()

        #     running_loss += loss.item()
        #     total += labels.shape[0]
        # scheduler.step()

        # test_acc = test(testloader)
        # if test_acc > best_acc:
        #     best_acc = test_acc
        #     torch.save(net.state_dict(), 'model/cifar10-resnet-%s-hard.pkl' % (args.algo))
        # print('[%3d] Train data = %d  Loss = %.4f  Test acc = %.4f  Time = %.2fs' % \
        #     (epoch + 1, total, running_loss / step, test_acc, time.time() - t))

        

        self.lr_scheduler.step()
        if (epoch + 1) % self.log_interval == 0:
            self._valid(self.valid_loader)


    def _valid(self, loader):
        total = 0
        correct = 0

        self.net.eval()
        with torch.no_grad():
            for data in loader:
                inputs = data[0].to(self.device)
                labels = data[1].to(self.device)

                outputs = self.net(inputs)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.shape[0]
                correct += predicted.eq(labels).sum().item()
        return correct / total


    def fit(self):
        for epoch in range(self.epochs):
            self._train_one_epoch(epoch)


    def evaluate(self):
        net_file = os.path.join(self.log_dir, 'net.pkl')
        assert os.path.exists(net_file), \
            'Assert Error: the best net does not exist'
        self.net.load_state_dict(torch.load(net_file))
        accuracy = self._valid(self.test_loader)
        # TODO: log


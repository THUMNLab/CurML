import os
import time
import torch

from ..datasets import get_dataset
from ..backbones import get_net
from ..utils import get_logger, set_random



class ImageClassifier():
    def __init__(self, data_name, net_name, 
                 device_name, random_seed,
                 algorithm_name, data_curriculum, 
                 model_curriculum, loss_curriculum):

        self.random_seed = random_seed

        self.data_curriculum = data_curriculum
        self.model_curriculum = model_curriculum
        self.loss_curriculum = loss_curriculum

        self._init_dataloader(data_name)
        self._init_model(net_name, device_name)
        self._init_logger(algorithm_name, data_name, 
                          net_name, random_seed)


    def _init_dataloader(self, data_name):
        set_random(self.random_seed)

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
            if torch.cuda.is_available() else 'cpu')
        self.net.to(self.device)

        self.epochs = 200
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.optimizer = torch.optim.SGD(
            self.net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4
        )
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs, eta_min=1e-6
        )

    
    def _init_logger(self, algorithm_name, data_name, 
                     net_name, random_seed):
        self.log_interval = 10

        # log_info = '%s-%s-%s-%d' % (
        #     algorithm_name, data_name, net_name, random_seed,
        # )
        log_info = '%s-%s-%s-%d-%s' % (
            algorithm_name, data_name, net_name, random_seed,
            time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        )
        self.log_dir = os.path.join('./runs', log_info)
        if not os.path.exists('./runs'): os.mkdir('./runs')
        if not os.path.exists(self.log_dir): os.mkdir(self.log_dir)
        else: print('The directory %s has already existed.' % (self.log_dir))
        
        log_file = os.path.join(self.log_dir, 'train.log')
        self.logger = get_logger(log_file)


    def _train(self):
        best_acc = 0.0

        for epoch in range(self.epochs):
            t = time.time()
            total = 0
            correct = 0
            train_loss = 0.0

            net = self.model_curriculum(self.net)            # curriculum part
            loader = self.data_curriculum(self.train_loader) # curriculum part

            net.train()
            for step, data in enumerate(loader):
                inputs = data[0].to(self.device)
                labels = data[1].to(self.device)
                indices = data[2].to(self.device)

                self.optimizer.zero_grad()
                outputs = net(inputs)
                loss = self.loss_curriculum(                 # curriculum part
                    self.criterion, outputs, labels, indices
                )
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(dim=1)
                correct += predicted.eq(labels).sum().item()
                total += labels.shape[0]
            
            self.lr_scheduler.step()
            self.logger.info(
                '[%3d] Train data = %5d  Loss = %.4f Train Acc = %.4f Time = %.2f'
                % (epoch + 1, total, train_loss / (step + 1), correct / total, time.time() - t))

            if (epoch + 1) % self.log_interval == 0:
                valid_acc = self._valid(self.valid_loader)
                if valid_acc > best_acc:
                    best_acc = valid_acc
                    torch.save(net.state_dict(), os.path.join(self.log_dir, 'net.pkl'))
                self.logger.info('[%3d] Valid data = %d Valid Acc = %.4f' 
                % (epoch + 1, len(self.valid_loader.dataset), valid_acc))
            


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
        set_random(self.random_seed)
        self._train()


    def evaluate(self):
        self._load_best_net()
        test_acc = self._valid(self.test_loader)
        self.logger.info('Final Test Acc = %.4f' % (test_acc))
        return test_acc


    def export(self):
        self._load_best_net()
        return self.net


    def _load_best_net(self):
        net_file = os.path.join(self.log_dir, 'net.pkl')
        assert os.path.exists(net_file), \
            'Assert Error: the net file does not exist'
        self.net.load_state_dict(torch.load(net_file))
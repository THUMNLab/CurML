import math
import torch
import torch.nn as nn
import numpy as np

from .base import BaseTrainer, BaseCL



class CoarseToFine(BaseCL):
    """
    
    Coarse-to-Fine Curriculum Learning. https://arxiv.org/pdf/2106.04072
    """
    def __init__(self, cluster_K, num_classes, pretrained_net):
        super(CoarseToFine, self).__init__()

        self.name = 'coarse_to_fine'

        self.epoch = 0
        self.classify_cnt = 0
        self.num_classes = num_classes
        self.cluster_K = cluster_K
        
        self.confusion_matrix = torch.zeros(self.num_classes ** 2)
        self.confusion_matrix = self.confusion_matrix.reshape(self.num_classes, self.num_classes)

        if self.cluster_K > self.num_classes:
            raise ValueError("The number of clusters should not exceed the number of classes.")

        self.pretrained_model = pretrained_net


    def data_prepare(self, loader):
        self.dataloader = loader
        self.dataset = self.CLDataset(loader.dataset)
        self.data_size = len(self.dataset)
        self.batch_size = loader.batch_size
        self.n_batches = (self.data_size - 1) // self.batch_size + 1
    

    def model_prepare(self, net, device, epochs, criterion, optimizer, lr_scheduler):
        self.device = device
        try:
            self.classifier = net.layer1
        except:
            raise ValueError("net should have a linear classifier")
        self.total_epoch = epochs
    

    def data_curriculum(self, loader):
        if self.epoch == 0:
            self._pretrain()
            self._affinity_clustering()

        self.epoch += 1
        if self.epoch > self.schedule[self.classify_cnt]:
            # transfer acquired knowledge from the last stage
            if isinstance(self.classifier, nn.Linear):
                self.classifier.out_features = self.num_cluster[self.classify_tot - 1 - self.classify_cnt]
                nn.init.uniform_(self.classifier.weight, -math.sqrt(1 / self.classifier.in_features), math.sqrt(1 / self.classifier.in_features))
                nn.init.uniform_(self.classifier.bias, -math.sqrt(1 / self.classifier.in_features), math.sqrt(1 / self.classifier.in_features))
            else:
                raise NotImplementedError()
            self.classify_cnt += 1
        return super().data_curriculum(loader)
    

    def loss_curriculum(self, criterion, outputs, labels, indices):
        cluster_label = torch.Tensor([self.classify_labels[self.classify_tot - self.classify_cnt][labels[index]] for index in range(len(labels))])
        cluster_label = cluster_label.type(torch.LongTensor).to(self.device)
        loss = torch.mean(criterion(outputs, cluster_label))
        return loss
    

    def _pretrain(self):
        # calculate confusion matrix
        
        for step, data in enumerate(self.dataloader):
            inputs = data[0].to(self.device)
            labels = data[1].to(self.device)
            with torch.no_grad():
                outputs = self.pretrained_model(inputs)
                _, predicted = torch.max(outputs, dim=1)
                for index in range(len(predicted)):
                    self.confusion_matrix[int(labels[index])][int(predicted[index])] += 1
                    
        
        for index in range(self.num_classes):
            self.confusion_matrix[index][index] = 0
            confusion_sum = torch.sum(self.confusion_matrix[index])
            self.confusion_matrix[index] /= confusion_sum

        self.confusion_matrix = 1 - self.confusion_matrix
        

    # Disjoint Set Union, using path compression
    def _find_fa(self, x): 
        if self.fa[x] == x:
            return x
        self.fa[x] = self._find_fa(self.fa[x])
        return self.fa[x]


    def _affinity_clustering(self):
        # use affinity clustering to cluster the data
        tot = self.num_classes
        self.num_cluster = np.array([tot])
        classify_labels = [[-1 for inner_index in range(tot)] for index in range(tot)]
        classify_labels[0] = [index for index in range(tot)]
        self.fa = np.arange(tot)
        epoch_cnt = 0
        while (tot > self.cluster_K):
            minimum_path = np.ones(tot)
            target = np.zeros(tot)
            for index_x in range(self.num_classes):
                label_x = classify_labels[epoch_cnt][index_x]
                for index_y in range(self.num_classes):
                    label_y = classify_labels[epoch_cnt][index_y]
                    if label_x == label_y:
                        continue
                    if self.confusion_matrix[index_x][index_y] <= minimum_path[label_x]:
                        minimum_path[label_x] = self.confusion_matrix[index_x][index_y]
                        target[label_x] = index_y

            for index in range(self.num_classes):
                label = classify_labels[epoch_cnt][index]
                self.fa[self._find_fa(int(index))] = self._find_fa(int(target[label]))

            for index in range(self.num_classes):
                self.fa[index] = self._find_fa(int(index))

            cnt = 0
            for index in range(self.num_classes):
                if classify_labels[epoch_cnt + 1][self.fa[index]] == -1:
                    classify_labels[epoch_cnt + 1][self.fa[index]] = cnt
                    cnt += 1
                classify_labels[epoch_cnt + 1][index] = classify_labels[epoch_cnt + 1][self.fa[index]]
            tot = cnt
            epoch_cnt += 1
            self.num_cluster = np.append(self.num_cluster, tot)
        
        classify = ()
        for index in range(epoch_cnt + 1):
            classify += (torch.Tensor([classify_labels[index],]),)
        self.classify_labels = torch.stack(classify, dim=1)[0]
        self.classify_tot = epoch_cnt + 1

        self._scheduler()

        print("classify = ", classify)
        print("curriculum schedule = ", self.schedule)

    def _scheduler(self):
        self.schedule = np.array([0])
        for index in range(self.classify_tot):
            self.schedule = np.append(self.schedule, np.floor(self.num_cluster[self.classify_tot - 1 - index] * self.total_epoch / np.sum(self.num_cluster)))
        self.schedule[len(self.schedule) - 1] = self.total_epoch


class CoarseToFineTrainer(BaseTrainer):
    def __init__(self, data_name, net_name, device_name, num_epochs, random_seed,
                 cluster_K, num_classes, pretrained_net):
        
        cl = CoarseToFine(cluster_K, num_classes, pretrained_net)

        super(CoarseToFineTrainer, self).__init__(
            data_name, net_name, device_name, num_epochs, random_seed, cl)

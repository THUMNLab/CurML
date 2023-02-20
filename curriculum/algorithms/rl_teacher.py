from collections import deque
import numpy as np
from torch.utils.data import DataLoader, Subset

from .base import BaseTrainer, BaseCL



def estimate_slope(x, y):
    assert len(x) == len(y)
    A = np.vstack([x, np.ones(len(x))]).T
    c, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return c



class EpsilonGreedyPolicy:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon


    def __call__(self, Q):
        # find the best action with random tie-breaking
        idx = np.where(Q == np.max(Q))[0]
        assert len(idx) > 0, str(Q)
        a = np.random.choice(idx)

        # create a probability distribution
        p = np.zeros(len(Q))
        p[a] = 1

        # Mix in a uniform distribution, to do exploration and
        # ensure we can compute slopes for all tasks
        p = p * (1.0 - self.epsilon) + self.epsilon / p.shape[0]

        assert np.isclose(np.sum(p), 1)
        return p



class ThompsonPolicy(EpsilonGreedyPolicy):
    pass



class RLTeacherOnline(BaseCL):
    """Reinforcement Learning Teacher CL Algorithm. 

    Teacher-student curriculum learning. https://arxiv.org/pdf/1707.00183
    """
    def __init__(self, ):
        super(RLTeacherOnline, self).__init__()

        self.name = 'rl_teacher_online'
        self.policy = EpsilonGreedyPolicy(0.01)

        self.catnum = 10
        self.alpha = 0.1
        self.total = [0 for i in range(self.catnum)]
        self.abs = False

        self.accs = [0 for i in range(self.catnum)]
        self.reward = []


    def data_split(self):
        self.indexs = [range(i * self.data_size // self.catnum, (i + 1) * self.data_size // self.catnum) 
                       for i in range(self.catnum)]
        l = [len(self.indexs[i]) // self.catnum for i in range(self.catnum)]
        self.data = [DataLoader(Subset(self.dataset, self.indexs[i]), self.batch_size, shuffle=True)
                     for i in range(self.catnum)]
        self.validationData = [DataLoader(Subset(self.dataset, self.indexs[i][:l[i]]), self.batch_size, shuffle=True)
                               for i in range(self.catnum)]


    def data_prepare(self, loader):
        self.dataset = loader.dataset
        self.data_size = len(self.dataset)
        self.batch_size = loader.batch_size
        self.n_batches = (self.data_size - 1) // self.batch_size + 1
        self.data_split()


    def model_prepare(self, net, device, epochs, criterion, optimizer, lr_scheduler):
        self.device = device
        self.criterion = criterion 
        self.model = net


    def data_curriculum(self, loader):
        acc = 0
        accs = []
        self.reward = []
        for i in range(self.catnum):
            acc = 0
            for (sample, label) in self.validationData[i]:
                sample = sample.to(self.device)
                label = label.to(self.device)
                out = self.model(sample)
                _, pred = out.max(1)
                num_correct = (pred == label).sum().item()
                acc += num_correct/len(self.validationData[i])
            accs.append(acc)
        for i, j in zip(accs, self.accs):
            self.reward.append(i-j)
        self.accs = accs

        for i in range(self.catnum):
            self.total[i] = self.total[i] *(1.0-self.alpha) + self.reward[i] * self.alpha
        # self.total[self.training] = self.total[self.training] * (1.0 - self.alpha) + self.reward * self.alpha
        p = self.policy(np.abs(self.total)if self.abs else self.total)
        temp = np.random.choice(range(self.catnum), p=p)
        data_loader = DataLoader(self.CLDataset(self.data[temp].dataset), self.batch_size, shuffle=True) 
        return data_loader



class RLTeacherNaive(BaseCL):
    def __init__(self, ):
        super(RLTeacherNaive, self).__init__()

        self.name = 'rl_teacher_naive'
        self.policy = EpsilonGreedyPolicy(0.01)

        self.catnum = 10
        self.alpha = 0.1
        self.total = [0 for i in range(self.catnum)]
        self.abs = False

        self.window_size = 10
        self.scores = []
        self.epoch_index = 0
        self.accs = [0 for i in range(self.catnum)]
        self.reward = []


    def data_split(self):
        self.indexs = [range(i*self.data_size//self.catnum, (i+1)*self.data_size//self.catnum) for i in range(self.catnum)]
        l = [len(self.indexs[i])// self.catnum for i in range(self.catnum)]
        self.data = [DataLoader(Subset(self.dataset, self.indexs[i]), self.batch_size, shuffle=True)for i in range(self.catnum)]
        self.validationData = [DataLoader(Subset(self.dataset, self.indexs[i][:l[i]]), self.batch_size, shuffle=True)for i in range(self.catnum)]


    def data_prepare(self, loader):
        self.dataset = loader.dataset
        self.data_size = len(self.dataset)
        self.batch_size = loader.batch_size
        self.n_batches = (self.data_size - 1) // self.batch_size + 1
        self.data_split()


    def model_prepare(self, net, device, epochs, criterion, optimizer, lr_scheduler):
        self.device = device
        self.criterion = criterion 
        self.model = net


    def data_curriculum(self, loader):
        acc = 0
        accs = []
        self.reward = []
        for i in range(self.catnum):
            acc = 0
            for (sample, label) in self.validationData[i]:
                sample = sample.to(self.device)
                label = label.to(self.device)
                out = self.model(sample)
                _, pred = out.max(1)
                num_correct = (pred == label).sum().item()
                acc += num_correct/len(self.validationData[i])
            accs.append(acc)
        for i, j in zip(accs, self.accs):
            self.reward.append(i-j)
        self.accs = accs

        self.scores.append(self.accs)
        if self.epoch_index % self.window_size == 0:
            self.reward = estimate_slope(range(len(self.scores)), self.scores)
            self.scores = []
            for i in range(self.catnum):
                self.total[i] = self.total[i] * (1.0 - self.alpha) + self.reward[i] * self.alpha
            p = self.policy(np.abs(self.total)if self.abs else self.total)
            self.training = np.random.choice(range(self.catnum), p=p)
            self.data_loader = DataLoader(self.CLDataset(self.data[self.training].dataset), self.batch_size, shuffle=True)
        self.epoch_index += 1

        return self.data_loader


class RLTeacherWindow(BaseCL):
    def __init__(self, ):
        super(RLTeacherWindow, self).__init__()

        self.name = 'rl_teacher_window'
        self.policy = EpsilonGreedyPolicy(0.01)

        # self.partnum = 10
        self.alpha = 0.1
        self.abs = False

        self.acc = 0
        self.training = 0
        self.reward = 0

        self.window_size = 10
        self.epoch_index = 1
        

    def split(self, data_loader, partnum):
        temp = data_loader.dataset
        k = len(temp)
        l = k // partnum
        self.data = []
        for i in range(partnum-1):
            self.data.append(Subset(temp, range(i * l, (i + 1) * l)))
        self.partnum = partnum - 1
        self.validationData = DataLoader(Subset(temp, range(self.partnum * l, k)), self.batch_size, shuffle=True)


    def data_prepare(self, loader):
        self.dataset = loader.dataset
        self.data_size = len(self.dataset)
        self.batch_size = loader.batch_size
        self.n_batches = (self.data_size - 1) // self.batch_size + 1
        self.split(loader, 10)
        self.total = np.zeros(self.partnum)
        self.scores = [deque(maxlen=self.window_size) for _ in range(self.partnum)]
        self.timesteps = [deque(maxlen=self.window_size) for _ in range(self.partnum)]


    def model_prepare(self, net, device, epochs, criterion, optimizer, lr_scheduler):
        self.device = device
        self.criterion = criterion 
        self.model = net


    def data_curriculum(self, loader):
        acc = 0
        for (sample, label) in self.validationData:
            sample = sample.to(self.device)
            label = label.to(self.device)
            out = self.model(sample)
            _, pred = out.max(1)
            num_correct = (pred == label).sum().item()
            acc += num_correct/len(self.validationData)
        self.reward = acc - self.acc
        self.acc = acc

        self.scores[self.training].append(self.acc)
        self.timesteps[self.training].append(self.epoch_index)
        self.total = [estimate_slope(timesteps, scores) if len(scores) > 1 else 1 for timesteps, scores in
                  zip(self.timesteps, self.scores)]
        p = self.policy(np.abs(self.total) if self.abs else self.total)
        self.training = np.random.choice(range(self.partnum), p=p)
        self.data_loader = DataLoader(self.CLDataset(self.data[self.training]), self.batch_size, shuffle=True)
        self.epoch_index += 1
        return self.data_loader



class RLTeacherSampling(BaseCL):
    def __init__(self, ):
        super(RLTeacherSampling, self).__init__()

        self.name = 'rl_teacher_sampling'
        self.policy = EpsilonGreedyPolicy(0.01)

        self.partnum = 10
        self.alpha = 0.1
        window_size = 10
        self.total = [0 for i in range(self.partnum)]
        self.abs = False

        self.accs = []
        self.prevr = np.zeros(self.partnum)
        self.training = 0

        self.window_size = window_size
        self.dscores = deque(maxlen=window_size)
        self.prevr = np.zeros(self.partnum)


    def data_prepare(self, loader):
        self.dataset = loader.dataset
        self.data_size = len(self.dataset)
        self.batch_size = loader.batch_size
        self.n_batches = (self.data_size - 1) // self.batch_size + 1
        partnum = 10
        temp = self.dataset
        k = len(temp)
        l = k // partnum
        self.data = []
        self.partnum = partnum
        for i in range(partnum-1):
            self.data.append(DataLoader(Subset(temp, range(i * l, (i + 1) * l)), self.batch_size, shuffle=True))
        self.data.append(DataLoader(Subset(temp, range((self.partnum - 1) * l, k)), self.batch_size, shuffle=True))


    def model_prepare(self, net, device, epochs, criterion, optimizer, lr_scheduler):
        self.device = device
        self.criterion = criterion 
        self.model = net


    def data_curriculum(self, loader):
        self.accs = []
        for i in range(self.partnum):
            acc = 0
            for (sample, label) in self.data[i]:
                sample = sample.to(self.device)
                label = label.to(self.device)
                out = self.model(sample)
                _, pred = out.max(1)
                num_correct = (pred == label).sum().item()
                acc += num_correct/len(self.data[i])
            self.accs.append(acc)
        if len(self.dscores) > 0:
            if isinstance(self.policy, ThompsonPolicy):
                slopes = [np.random.choice(drs) for drs in np.array(self.dscores).T]
            else:
                slopes = np.mean(self.dscores, axis=0)
        else:
            slopes = np.ones(self.partnum)
        p = self.policy(np.abs(slopes) if self.abs else slopes)
        self.training = np.random.choice(range(self.partnum), p=p)
        data_loader = DataLoader(self.CLDataset(self.data[self.training].dataset), self.batch_size, shuffle=True)
        dr = [i-j for i, j in zip(self.accs, self.prevr)]
        self.prevr = self.accs
        self.dscores.append(dr)
        return data_loader



class RLTeacherTrainer(BaseTrainer):
    def __init__(self, data_name, net_name, device_name, 
                 num_epochs, random_seed, policy):
        
        cl_dict = {'online': RLTeacherOnline,
                   'naive': RLTeacherNaive,
                   'window': RLTeacherWindow,
                   'sampling': RLTeacherSampling}
        cl = cl_dict[policy]()

        super(RLTeacherTrainer, self).__init__(
            data_name, net_name, device_name, num_epochs, random_seed, cl)

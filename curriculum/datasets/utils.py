import random
import numpy as np
import torch
from torch.utils.data import Dataset



class LabelNoise(Dataset):
    def __init__(self, dataset, noise_ratio, num_labels):
        self.dataset = dataset
        self.noise_ratio = noise_ratio
        self.num_labels = num_labels
        self.labels = []
        for i, (_, y) in enumerate(self.dataset):
            if random.random() < self.noise_ratio:
                self.labels.append(
                    random.choice(list(range(0, y)) + list(range(y + 1, num_labels))))
            else:
                self.labels.append(y)

    def __getitem__(self, index):
        return (self.dataset[index][0], self.labels[index])

    def __len__(self):
        return len(self.dataset)



class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask

        return img



# TODO: Lighting
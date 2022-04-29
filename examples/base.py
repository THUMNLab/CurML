import argparse

from curriculum.algorithms import BaseTrainer



# input hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--epochs', type=int, default='200')
parser.add_argument('--seed', type=int, default='42')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--algo', type=str, default='base')
args = parser.parse_args()


# train and evaluate
trainer = BaseTrainer(
     dataset='cifar10',
     backbone='resnet18',
     # TODO
)
trainer.fit()
trainer.evaluate()
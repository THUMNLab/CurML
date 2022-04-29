import argparse

from curriculum.datasets import get_cifar10_dataset


# input hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--data_dir', type=str, default='../data')
parser.add_argument('--epochs', type=int, default='200')
parser.add_argument('--seed', type=int, default='42')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--algo', type=str, default='baseline')
args = parser.parse_args()
info = 'epochs = %d, seed = %d, algo = %s' \
     % (args.epochs, args.seed, args.algo)
print(info)


train_dataset, valid_dataset, test_dataset = get_cifar10_dataset(args.data_dir)

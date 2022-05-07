import argparse

from curriculum.algorithms import BaseTrainer


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='cifar10')
parser.add_argument('--net', type=str, default='resnet')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()


trainer = BaseTrainer(
     data_name=args.data,
     net_name=args.net,
     device_name=args.device,
     num_epochs=args.epochs,
     random_seed=args.seed,
)
trainer.fit()
trainer.evaluate()
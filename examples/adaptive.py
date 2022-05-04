import argparse

from curriculum.algorithms import AdaptiveTrainer



parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='cifar10')
parser.add_argument('--net', type=str, default='resnet')
parser.add_argument('--seed', type=int, default='42')
parser.add_argument('--device', type=str, default='cuda')
args = parser.parse_args()


trainer = AdaptiveTrainer(
    data_name=args.data,
    net_name=args.net,
    device_name=args.device,
    random_seed=args.seed,
)
trainer.fit()
trainer.evaluate()
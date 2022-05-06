import argparse

from curriculum.algorithms import LocalToGlobalTrainer



parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='cifar10')
parser.add_argument('--net', type=str, default='resnet')
parser.add_argument('--seed', type=int, default='42')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--start_size', type=int, default=0)
parser.add_argument('--grow_size', type=int, default=2)
parser.add_argument('--grow_interval', type=int, default=3)
parser.add_argument('--strategy', type=str, default='random')
args = parser.parse_args()


trainer = LocalToGlobalTrainer(
    data_name=args.data,
    net_name=args.net,
    device_name=args.device,
    random_seed=args.seed,
    start_size = args.start_size, 
    grow_size=args.grow_size, 
    grow_interval=args.grow_interval, 
    strategy=args.strategy,
)
trainer.fit()
trainer.evaluate()
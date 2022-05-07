import argparse

from curriculum.algorithms import CBSTrainer



parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='cifar10')
parser.add_argument('--net', type=str, default='resnet')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--kernel_size', type=int, default=3)
parser.add_argument('--start_std', type=float, default=1.0)
parser.add_argument('--grow_factor', type=float, default=0.9)
parser.add_argument('--grow_interval', type=int, default=5)

args = parser.parse_args()


trainer = CBSTrainer(
    data_name=args.data,
    net_name=args.net,
    device_name=args.device,
    num_epochs=args.epochs,
    random_seed=args.seed,
    kernel_size=args.kernel_size,
    start_std=args.start_std,
    grow_factor=args.grow_factor,
    grow_interval=args.grow_interval,
)
trainer.fit()
trainer.evaluate()
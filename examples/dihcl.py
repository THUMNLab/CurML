import argparse

from curriculum.algorithms import DIHCLTrainer



parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='cifar10')
parser.add_argument('--net', type=str, default='resnet')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--warm_epoch', type=int, default=10)
parser.add_argument('--discount_factor', type=float, default=0.9)
parser.add_argument('--decay_rate', type=float, default=0.85)
parser.add_argument('--bottom_size', type=float, default=0.2)
parser.add_argument('--type', type=str, default='loss')
parser.add_argument('--sample_type', type=str, default='rand')
parser.add_argument('--cei', type=float, default=None)
args = parser.parse_args()


trainer = DIHCLTrainer(
    data_name=args.data,
    net_name=args.net,
    device_name=args.device,
    num_epochs=args.epochs,
    random_seed=args.seed,
    warm_epoch=args.warm_epoch,
    discount_factor=args.discount_factor,
    decay_rate=args.decay_rate,
    bottom_size=args.bottom_size,
    type=args.type,
    sample_type=args.sample_type,
    cei=args.cei,
)
trainer.fit()
trainer.evaluate()
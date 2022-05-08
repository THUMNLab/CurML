import argparse

from curriculum.algorithms import MetaWeightNetTrainer, MetaWeightNetTrainer_2



parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='cifar10')
parser.add_argument('--net', type=str, default='resnet')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--type', type=int, default=1)
args = parser.parse_args()

if args.type == 1:
    trainer = MetaWeightNetTrainer(
        data_name=args.data,
        net_name=args.net,
        device_name=args.device,
        num_epochs=args.epochs,
        random_seed=args.seed,
    )
elif  args.type == 2:
    trainer = MetaWeightNetTrainer_2(
        data_name=args.data,
        net_name=args.net,
        device_name=args.device,
        num_epochs=args.epochs,
        random_seed=args.seed,
    )
else:
    raise TypeError

trainer.fit()
trainer.evaluate()
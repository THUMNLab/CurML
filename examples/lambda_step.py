import argparse

from curriculum.algorithms import LambdaStepTrainer



parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='cifar10')
parser.add_argument('--net', type=str, default='resnet')
parser.add_argument('--seed', type=int, default='42')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--start_rate', type=float, default=0.0)
parser.add_argument('--grow_epochs', type=int, default=200)
parser.add_argument('--grow_fn', type=str, default='linear')
args = parser.parse_args()


trainer = LambdaStepTrainer(
    data_name=args.data,
    net_name=args.net,
    device_name=args.device,
    random_seed=args.seed,
    start_rate=args.start_rate,
    grow_epochs=args.grow_epochs,
    grow_fn=args.grow_fn,
)
trainer.fit()
trainer.evaluate()
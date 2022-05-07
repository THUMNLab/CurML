import argparse

from curriculum.algorithms import MinimaxTrainer



parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='cifar10')
parser.add_argument('--net', type=str, default='resnet')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--schedule_epoch', type=int, default=20)
parser.add_argument('--warm_epoch', type=int, default=5)
parser.add_argument('--lam', type=float, default=1)
parser.add_argument('--minlam', type=float, default=0.2)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--delta', type=float, default=None)
parser.add_argument('--initial_size', type=float, default=None)
parser.add_argument('--fe_alpha', type=float, default=2)
parser.add_argument('--fe_beta', type=float, default=0.75)
parser.add_argument('--fe_gamma', type=float, default=0.9)
parser.add_argument('--fe_lambda', type=float, default=0.9)
parser.add_argument('--fe_entropy', type=bool, default=False)
parser.add_argument('--fe_gsrow', type=bool, default=False)
parser.add_argument('--fe_central_op', type=bool, default=True)
parser.add_argument('--fe_central_min', type=bool, default=False)
parser.add_argument('--fe_central_sum', type=bool, default=False)
parser.add_argument('--num_classes', type=int, default=10)
args = parser.parse_args()


trainer = MinimaxTrainer(
    data_name=args.data,
    net_name=args.net,
    device_name=args.device,
    num_epochs=args.epochs,
    random_seed=args.seed,
    schedule_epoch=args.schedule_epoch,
    warm_epoch=args.warm_epoch,
    lam=args.lam,
    minlam=args.minlam,
    gamma=args.gamma,
    delta=args.delta,
    initial_size=args.initial_size,
    fe_alpha=args.fe_alpha,
    fe_beta=args.fe_beta,
    fe_gamma=args.fe_gamma,
    fe_lambda=args.fe_lambda,
    fe_entropy=args.fe_entropy,
    fe_gsrow=args.fe_gsrow,
    fe_central_op=args.fe_central_op,
    fe_central_min=args.fe_central_min,
    fe_central_sum=args.fe_central_sum,
    num_classes=args.num_classes,
)
trainer.fit()
trainer.evaluate()
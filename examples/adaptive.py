import argparse

from curriculum.algorithms import \
    BaseTrainer, AdaptiveTrainer



parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='cifar10')
parser.add_argument('--net', type=str, default='resnet')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--pace_p', type=float, default=0.1)
parser.add_argument('--pace_q', type=float, default=1.2)
parser.add_argument('--pace_r', type=int, default=15)
parser.add_argument('--inv', type=int, default=20)
parser.add_argument('--alpha', type=float, default=0.7)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--gamma_decay', type=float, default=None)
parser.add_argument('--bottom_gamma', type=float, default=0.1)
parser.add_argument('--teacher_dir', type=str, default=None)
args = parser.parse_args()


pretrainer = BaseTrainer(
    data_name=args.data,
    net_name=args.net,
    device_name=args.device,
    num_epochs=args.epochs,
    random_seed=42,
)
if args.teacher_dir is None:
    pretrainer.fit()
pretrainer.evaluate(args.teacher_dir)
teacher_net = pretrainer.export(args.teacher_dir)


trainer = AdaptiveTrainer(
    data_name=args.data,
    net_name=args.net,
    device_name=args.device,
    num_epochs=args.epochs,
    random_seed=args.seed,
    num_classes=args.num_classes,
    pace_p=args.pace_p,
    pace_q=args.pace_q,
    pace_r=args.pace_r,
    inv=args.inv,
    alpha=args.alpha,
    gamma=args.gamma,
    gamma_decay=args.gamma_decay,
    bottom_gamma=args.bottom_gamma,
    pretrained_net=teacher_net,
)
trainer.fit()
trainer.evaluate()
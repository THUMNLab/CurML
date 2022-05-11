import argparse

from curriculum.algorithms import \
    BaseTrainer, TransferTeacherTrainer



parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='cifar10')
parser.add_argument('--net', type=str, default='resnet')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--start_rate', type=float, default=0.0)
parser.add_argument('--grow_epochs', type=int, default=200)
parser.add_argument('--grow_fn', type=str, default='linear')
parser.add_argument('--weight_fn', type=str, default='hard')
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


trainer = TransferTeacherTrainer(
    data_name=args.data,
    net_name=args.net,
    device_name=args.device,
    num_epochs=args.epochs,
    random_seed=args.seed,
    start_rate=args.start_rate,
    grow_epochs=args.grow_epochs,
    grow_fn=args.grow_fn,
    weight_fn=args.weight_fn,
    teacher_net=teacher_net,
)
trainer.fit()
trainer.evaluate()
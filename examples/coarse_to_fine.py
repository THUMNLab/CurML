import argparse

from curriculum.algorithms import \
    BaseTrainer, CoarseToFineTrainer



parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='cifar10')
parser.add_argument('--net', type=str, default='resnet')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--cluster_K', type=int, default=3)
parser.add_argument('--num_classes', type=int, default=10)
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


trainer = CoarseToFineTrainer(
    data_name=args.data,
    net_name=args.net,
    device_name=args.device,
    num_epochs=args.epochs,
    random_seed=args.seed,
    cluster_K=args.cluster_K,
    num_classes=args.num_classes,
    pretrained_net=teacher_net,
)
trainer.fit()
trainer.evaluate()
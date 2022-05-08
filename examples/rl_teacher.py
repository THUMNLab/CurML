import argparse

from curriculum.algorithms import RLTeacherTrainer_1, RLTeacherTrainer_2, RLTeacherTrainer_3, RLTeacherTrainer_4



parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='cifar10')
parser.add_argument('--net', type=str, default='resnet')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--type', type=int, default=1)
args = parser.parse_args()

if args.type == 1:
    trainer = RLTeacherTrainer_1(
        data_name=args.data,
        net_name=args.net,
        device_name=args.device,
        num_epochs=args.epochs,
        random_seed=args.seed,
    )
elif args.type == 2:
    trainer = RLTeacherTrainer_2(
        data_name=args.data,
        net_name=args.net,
        device_name=args.device,
        num_epochs=args.epochs,
        random_seed=args.seed,
    )
elif args.type == 3:
    trainer = RLTeacherTrainer_3(
        data_name=args.data,
        net_name=args.net,
        device_name=args.device,
        num_epochs=args.epochs,
        random_seed=args.seed,
    )
elif args.type == 4:
    trainer = RLTeacherTrainer_4(
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
import math
import argparse

from curriculum.trainers import ImageClassifier
from curriculum.algorithms import BabyStep, Superloss



parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='cifar10')
parser.add_argument('--net', type=str, default='resnet')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--start_rate', type=float, default=0.0)
parser.add_argument('--grow_rate', type=float, default=0.1)
parser.add_argument('--grow_interval', type=int, default=20)
parser.add_argument('--not_sorted', action="store_true")
parser.add_argument('--tau', type=float, default=math.log(10))
parser.add_argument('--lam', type=float, default=1.0)
parser.add_argument('--fac', type=float, default=0.0)
args = parser.parse_args()


baby_step = BabyStep(
    start_rate=args.start_rate,
    grow_rate=args.grow_rate,
    grow_interval=args.grow_interval,
    not_sorted=args.not_sorted,
)

superloss = Superloss(
    tau=args.tau,
    lam=args.lam,
    fac=args.fac,
)

trainer = ImageClassifier(
    data_name=args.data,
    net_name=args.net,
    device_name=args.device,
    num_epochs=args.epochs,
    random_seed=args.seed,
    algorithm_name='babystep+superloss',
    data_prepare=baby_step.data_prepare,
    model_prepare=superloss.model_prepare,
    data_curriculum=baby_step.data_curriculum, 
    model_curriculum=superloss.model_curriculum, 
    loss_curriculum=superloss.loss_curriculum
)
trainer.fit()
trainer.evaluate()
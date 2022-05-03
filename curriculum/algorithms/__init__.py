from .base import BaseCL, BaseTrainer

from .baby_step import BabyStep, BabyStepTrainer
from .lambda_step import LambdaStep, LambdaStepTrainer
from .self_paced import SelfPaced, SelfPacedTrainer
from .transfer_teacher import TransferTeacher, TransferTeacherTrainer
from .superloss import Superloss, SuperlossTrainer



all = [
    'BaseCL'
    'BaseTrainer',

    'BabyStep',
    'BabyStepTrainer',

    'LambdaStep',
    'LambdaStepTrainer',

    'SelfPaced',
    'SelfPacedTrainer',

    'TransferTeacher',
    'TransferTeacherTrainer',

    'Superloss',
    'SuperlossTrainer',
]
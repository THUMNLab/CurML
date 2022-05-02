from .base import BaseTrainer

from .baby_step import BabyStepTrainer
from .lambda_step import LambdaStepTrainer
from .self_paced import SelfPacedTrainer
from .transfer_teacher import TransferTeacherTrainer
from .superloss import SuperlossTrainer



all = [
    'BaseTrainer',

    'BabyStepTrainer',
    'LambdaStepTrainer',
    'SelfPacedTrainer',
    'TransferTeacherTrainer',
    'SuperlossTrainer',
]
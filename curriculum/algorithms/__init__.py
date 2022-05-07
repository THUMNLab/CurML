from .base import BaseCL, BaseTrainer

from .baby_step import BabyStep, BabyStepTrainer
from .lambda_step import LambdaStep, LambdaStepTrainer
from .self_paced import SelfPaced, SelfPacedTrainer
from .transfer_teacher import TransferTeacher, TransferTeacherTrainer
from .superloss import Superloss, SuperlossTrainer
from .data_parameters import DataParameters, DataParametersTrainer
from .local_to_global import LocalToGlobal, LocalToGlobalTrainer
from .dihcl import DIHCL, DIHCLTrainer
from .cbs import CBS, CBSTrainer
from .minimax import Minimax, MinimaxTrainer
from .adaptive import Adaptive, AdaptiveTrainer
from .coarse_to_fine import CoarseToFine, CoarseToFineTrainer
from .rl_teacher import RLTeacher, RLTeacherTrainer
from .screener_net import ScreenerNet, ScreenerNetTrainer
from .meta_weight_net import MetaWeightNet, MetaWeightNetTrainer
from .meta_reweight import MetaReweight, MetaReweightTrainer



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

    'DataParameters',
    'DataParametersTrainer',

    'LocalToGlobal',
    'LocalToGlobalTrainer',

    'DIHCL',
    'DIHCLTrainer',

    'CBS',
    'CBSTrainer',

    'Minimax',
    'MinimaxTrainer',

    'Adaptive',
    'AdaptiveTrainer',

    'CoarseToFine',
    'CoarseToFineTrainer',

    'RLTeacher',
    'RLTeacherTrainer',

    'ScreenerNet',
    'ScreenerNetTrainer',

    'MetaWeightNet',
    'MetaWeightNetTrainer',

    'MetaReweight',
    'MetaReweightTrainer',
]
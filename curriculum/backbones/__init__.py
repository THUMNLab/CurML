# The code is developed based on
# https://github.com/kuangliu/pytorch-cifar

from .vgg import *
from .dpn import *
from .lenet import *
from .senet import *
from .pnasnet import *
from .densenet import *
from .googlenet import *
from .shufflenet import *
from .shufflenetv2 import *
from .resnet import *
from .resnext import *
from .preact_resnet import *
from .mobilenet import *
from .mobilenetv2 import *
from .efficientnet import *
from .regnet import *
from .dla_simple import *
from .dla import *



def get_model(model_name):

    model_dict = {
        'vgg': VGG19,
        'resnet': ResNet18,
        'preactresnet': PreActResNet18,
        'googlenet': GoogLeNet,
        'densenet': DenseNet121,
        'resnext': ResNeXt29_2x64d,
        'mobilenet': MobileNet,
        'dpn': DPN92,
        'shufflenet': ShuffleNetG2,
        'senet': SENet18,
        'efficientnet': EfficientNetB0,
        'regnetx': RegNetX_200MF,
        # TODO: more version of the nets above
    }

    assert model_name in model_dict, \
        'Assert Error: model_name should be in ' + str(list(model_dict.keys()))
    
    return model_dict[model_name]()
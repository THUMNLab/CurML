# The code is developed based on
# https://github.com/kuangliu/pytorch-cifar

from .convnet import *
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



def get_net(net_name, data_name):
    net_dict = {
        'convnet': ConvNet,
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

    assert net_name in net_dict, \
        'Assert Error: net_name should be in ' + str(list(net_dict.keys()))

    classes_dict = {
        'cifar10': 10, 
        'cifar100': 100,
    }
    
    return net_dict[net_name](num_classes=classes_dict[data_name])
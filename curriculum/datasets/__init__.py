from .utils import Cutout

from .cifar10 import get_cifar10_dataset
# TODO: other image datasets like cifar100, imagenet, etc.
# TODO: other text  datasets like ptb, wikitext, etc.



all = [
    'Cutout',

    'get_cifar10_dataset',
]



# TODO: function: choose dataset based on its name
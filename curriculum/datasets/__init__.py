from .utils import Cutout

from .cifar10 import get_cifar10_dataset
# TODO: other image datasets like cifar100, imagenet, etc.
# TODO: other text  datasets like ptb, wikitext, etc.



all = [
    'Cutout',

    'get_cifar10_dataset',
]


# TODO: function: choose dataset based on its name
def get_dataset(data_dir, data_name):

    data_dict = {
        'cifar10': get_cifar10_dataset
    }

    assert data_name in data_dict, \
        'Assert Error: data_name should be in ' + str(list(data_dict.keys()))
    
    return data_dict[data_name](data_dir)
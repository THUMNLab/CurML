from torch.utils.data import DataLoader

from curriculum.datasets import get_cifar10_dataset



train_dataset, valid_dataset, test_dataset = get_cifar10_dataset(
    data_dir='../../data', valid_ratio=0.1, shuffle=True,
    random_seed=43, augment=True, cutout_length=0
)

train_loader = DataLoader(
    train_dataset, batch_size=128, shuffle=True,
    num_workers=4, pin_memory=True,
)
valid_loader = DataLoader(
    valid_dataset, batch_size=128, shuffle=True,
    num_workers=4, pin_memory=True,
)
test_loader = DataLoader(
    test_dataset, batch_size=128, shuffle=False,
    num_workers=4, pin_memory=True,
)
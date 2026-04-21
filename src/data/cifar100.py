import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset


_CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
_CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def get_cifar100_loaders(
    data_root: str,
    batch_size: int,
    num_workers: int = 0,
    smoke_test: bool = False,
):
    """
    Returns CIFAR-100 DataLoaders.

    Args:
        data_root:   Directory to download / read CIFAR-100 from.
        batch_size:  Mini-batch size.
        num_workers: DataLoader worker processes.
        smoke_test:  If True, returns tiny synthetic datasets (200 train, 50 test).

    Returns:
        train_loader, test_loader, dataset_train, dataset_test
    """
    if smoke_test:
        dataset_train = TensorDataset(
            torch.randn(200, 3, 32, 32), torch.randint(0, 100, (200,))
        )
        dataset_test = TensorDataset(
            torch.randn(50, 3, 32, 32), torch.randint(0, 100, (50,))
        )
        train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader, dataset_train, dataset_test

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR100_MEAN, _CIFAR100_STD),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR100_MEAN, _CIFAR100_STD),
    ])

    dataset_train = torchvision.datasets.CIFAR100(
        root=data_root, train=True, download=True, transform=transform_train
    )
    dataset_test = torchvision.datasets.CIFAR100(
        root=data_root, train=False, download=True, transform=transform_test
    )
    train_loader = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader, dataset_train, dataset_test
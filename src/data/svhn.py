"""
SVHN dataset loader for OOD evaluation.

Images are normalized with the ID dataset's (CIFAR-10 or CIFAR-100) mean/std
so that the model's input distribution is aligned.
"""

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD = (0.2470, 0.2435, 0.2616)

_CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
_CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def get_svhn_loader(
    data_root: str,
    batch_size: int,
    num_workers: int = 0,
    id_normalization: str = "cifar10",
    split: str = "test",
) -> DataLoader:
    """
    Returns a DataLoader for SVHN, normalized with the ID dataset's mean/std.

    Args:
        data_root:        Directory to download / read SVHN from.
        batch_size:       Mini-batch size.
        num_workers:      DataLoader worker processes.
        id_normalization: "cifar10" or "cifar100".
        split:            "test" (default), "train", or "extra".

    Returns:
        DataLoader yielding (images, labels).
    """
    if id_normalization == "cifar100":
        mean, std = _CIFAR100_MEAN, _CIFAR100_STD
    else:
        mean, std = _CIFAR10_MEAN, _CIFAR10_STD

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    dataset = torchvision.datasets.SVHN(
        root=data_root,
        split=split,
        download=True,
        transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
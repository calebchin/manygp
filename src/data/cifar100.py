import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset


_CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
_CIFAR100_STD = (0.2675, 0.2565, 0.2761)


class TwoCropTransform:
    """Generate two independently augmented views of the same image."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, image):
        return torch.stack(
            [self.base_transform(image), self.base_transform(image)],
            dim=0,
        )


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
        (dataset_train / dataset_test expose .targets for evaluation)
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


def get_cifar100_supcon_loaders(
    data_root: str,
    batch_size: int,
    num_workers: int = 0,
    smoke_test: bool = False,
):
    """
    Returns CIFAR-100 loaders for supervised contrastive training and k-NN eval.

    The training loader yields `(views, labels)` where `views` has shape
    `(batch_size, 2, 3, 32, 32)`. Evaluation loaders use deterministic
    preprocessing for stable embedding extraction.
    """
    if smoke_test:
        train_images = torch.randn(200, 3, 32, 32)
        train_labels = torch.randint(0, 100, (200,))
        test_images = torch.randn(50, 3, 32, 32)
        test_labels = torch.randint(0, 100, (50,))

        train_contrastive = TensorDataset(
            train_images.unsqueeze(1).repeat(1, 2, 1, 1, 1), train_labels
        )
        train_eval = TensorDataset(train_images, train_labels)
        test_eval = TensorDataset(test_images, test_labels)

        train_loader = DataLoader(train_contrastive, batch_size=batch_size, shuffle=True)
        memory_loader = DataLoader(train_eval, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(test_eval, batch_size=batch_size, shuffle=False)
        return train_loader, memory_loader, val_loader, train_contrastive, test_eval

    contrastive_transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
            p=0.8,
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR100_MEAN, _CIFAR100_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR100_MEAN, _CIFAR100_STD),
    ])

    train_contrastive_dataset = torchvision.datasets.CIFAR100(
        root=data_root,
        train=True,
        download=True,
        transform=TwoCropTransform(contrastive_transform),
    )
    train_eval_dataset = torchvision.datasets.CIFAR100(
        root=data_root,
        train=True,
        download=True,
        transform=eval_transform,
    )
    test_eval_dataset = torchvision.datasets.CIFAR100(
        root=data_root,
        train=False,
        download=True,
        transform=eval_transform,
    )

    train_loader = DataLoader(
        train_contrastive_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    memory_loader = DataLoader(
        train_eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        test_eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, memory_loader, val_loader, train_contrastive_dataset, test_eval_dataset

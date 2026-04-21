import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset


_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class TwoCropTransform:
    """Generate two independently augmented views of the same image."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, image):
        return torch.stack(
            [self.base_transform(image), self.base_transform(image)],
            dim=0,
        )


class CIFAR10TwoViewDataset(torchvision.datasets.CIFAR10):
    """CIFAR-10 dataset that returns two independently augmented views and one label."""

    def __init__(self, *args, base_transform=None, **kwargs):
        if base_transform is None:
            raise ValueError("base_transform must be provided for CIFAR10TwoViewDataset")
        super().__init__(*args, transform=TwoCropTransform(base_transform), **kwargs)


def get_cifar10_eval_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
    ])


def get_cifar10_standard_train_transform():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
    ])


def get_cifar10_supcon_train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
            p=0.8,
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
    ])


def get_cifar10_loaders(
    data_root: str,
    batch_size: int,
    num_workers: int = 0,
    smoke_test: bool = False,
):
    """
    Returns CIFAR-10 DataLoaders.

    Args:
        data_root:   Directory to download / read CIFAR-10 from.
        batch_size:  Mini-batch size.
        num_workers: DataLoader worker processes.
        smoke_test:  If True, returns tiny synthetic datasets (200 train, 50 test).

    Returns:
        train_loader, test_loader, dataset_train, dataset_test
        (dataset_train / dataset_test expose .targets for evaluation)
    """
    if smoke_test:
        dataset_train = TensorDataset(
            torch.randn(200, 3, 32, 32), torch.randint(0, 10, (200,))
        )
        dataset_test = TensorDataset(
            torch.randn(50, 3, 32, 32), torch.randint(0, 10, (50,))
        )
        train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader, dataset_train, dataset_test

    transform_train = get_cifar10_standard_train_transform()
    transform_test = get_cifar10_eval_transform()

    dataset_train = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform_train
    )
    dataset_test = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform_test
    )
    train_loader = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader, dataset_train, dataset_test


def get_cifar10_supcon_loaders(
    data_root: str,
    batch_size: int,
    num_workers: int = 0,
    smoke_test: bool = False,
):
    """
    Returns CIFAR-10 loaders for supervised contrastive training and k-NN eval.

    The training loader yields `(views, labels)` where `views` has shape
    `(batch_size, 2, 3, 32, 32)`. Evaluation loaders use deterministic
    preprocessing for stable embedding extraction.
    """
    if smoke_test:
        train_images = torch.randn(200, 3, 32, 32)
        train_labels = torch.randint(0, 10, (200,))
        test_images = torch.randn(50, 3, 32, 32)
        test_labels = torch.randint(0, 10, (50,))

        train_contrastive = TensorDataset(
            train_images.unsqueeze(1).repeat(1, 2, 1, 1, 1), train_labels
        )
        train_eval = TensorDataset(train_images, train_labels)
        test_eval = TensorDataset(test_images, test_labels)

        train_loader = DataLoader(train_contrastive, batch_size=batch_size, shuffle=True)
        memory_loader = DataLoader(train_eval, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(test_eval, batch_size=batch_size, shuffle=False)
        return train_loader, memory_loader, val_loader, train_contrastive, test_eval

    contrastive_transform = get_cifar10_supcon_train_transform()
    eval_transform = get_cifar10_eval_transform()

    train_contrastive_dataset = CIFAR10TwoViewDataset(
        root=data_root,
        train=True,
        download=True,
        base_transform=contrastive_transform,
    )
    train_eval_dataset = torchvision.datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=eval_transform,
    )
    test_eval_dataset = torchvision.datasets.CIFAR10(
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


def get_cifar10_two_view_classification_loaders(
    data_root: str,
    batch_size: int,
    num_workers: int = 0,
    smoke_test: bool = False,
):
    """
    Returns CIFAR-10 loaders for classification training with two augmented views.

    The training loader yields `(views, labels)` where `views` has shape
    `(batch_size, 2, 3, 32, 32)`, using the same augmentation family as
    supervised contrastive training. Evaluation uses deterministic preprocessing.
    """
    if smoke_test:
        train_images = torch.randn(200, 3, 32, 32)
        train_labels = torch.randint(0, 10, (200,))
        test_images = torch.randn(50, 3, 32, 32)
        test_labels = torch.randint(0, 10, (50,))

        train_dataset = TensorDataset(
            train_images.unsqueeze(1).repeat(1, 2, 1, 1, 1), train_labels
        )
        val_dataset = TensorDataset(test_images, test_labels)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, train_dataset, val_dataset

    train_dataset = CIFAR10TwoViewDataset(
        root=data_root,
        train=True,
        download=True,
        base_transform=get_cifar10_supcon_train_transform(),
    )
    val_dataset = torchvision.datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=get_cifar10_eval_transform(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, train_dataset, val_dataset

import random
from contextlib import contextmanager

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Subset, TensorDataset


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


@contextmanager
def _seeded_random_state(seed: int):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    try:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)


class CIFAR10TwoViewDataset(torchvision.datasets.CIFAR10):
    """CIFAR-10 dataset that returns two independently augmented views and one label."""

    def __init__(self, *args, base_transform=None, **kwargs):
        if base_transform is None:
            raise ValueError("base_transform must be provided for CIFAR10TwoViewDataset")
        super().__init__(*args, transform=TwoCropTransform(base_transform), **kwargs)


class FixedAugmentTwoCropTransform:
    """Generate two deterministic augmented views keyed by dataset index."""

    def __init__(self, base_transform, base_seed: int = 0):
        self.base_transform = base_transform
        self.base_seed = base_seed

    def __call__(self, image, index: int):
        views = []
        for view_idx in range(2):
            seed = self.base_seed + index * 2 + view_idx
            with _seeded_random_state(seed):
                views.append(self.base_transform(image.copy()))
        return torch.stack(views, dim=0)


class CIFAR10DeterministicTwoViewDataset(torchvision.datasets.CIFAR10):
    """CIFAR-10 dataset with two fixed augmented views per image."""

    def __init__(self, *args, base_transform=None, base_seed: int = 0, **kwargs):
        if base_transform is None:
            raise ValueError("base_transform must be provided for CIFAR10DeterministicTwoViewDataset")
        super().__init__(*args, transform=None, **kwargs)
        self.transform = FixedAugmentTwoCropTransform(base_transform, base_seed=base_seed)

    def __getitem__(self, index):
        image, target = self.data[index], self.targets[index]
        image = Image.fromarray(image)
        views = self.transform(image, index)
        return views, target


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
    val_split: float = 0.1,
):
    """
    Returns CIFAR-10 DataLoaders with a train / val / test split.

    The training set (50 000 samples) is split into a training subset and a
    validation subset using a fixed random seed so results are reproducible.
    The held-out test set (10 000 samples) is never used for checkpoint
    selection.

    Args:
        data_root:   Directory to download / read CIFAR-10 from.
        batch_size:  Mini-batch size.
        num_workers: DataLoader worker processes.
        smoke_test:  If True, returns tiny synthetic datasets (160/40/50).
        val_split:   Fraction of the 50K training set to use as validation.

    Returns:
        train_loader, val_loader, test_loader,
        dataset_train, dataset_val, dataset_test
    """
    if smoke_test:
        n_train, n_val, n_test = 160, 40, 50
        dataset_train = TensorDataset(
            torch.randn(n_train, 3, 32, 32), torch.randint(0, 10, (n_train,))
        )
        dataset_val = TensorDataset(
            torch.randn(n_val, 3, 32, 32), torch.randint(0, 10, (n_val,))
        )
        dataset_test = TensorDataset(
            torch.randn(n_test, 3, 32, 32), torch.randint(0, 10, (n_test,))
        )
        train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader, dataset_train, dataset_val, dataset_test

    transform_train = get_cifar10_standard_train_transform()
    transform_eval = get_cifar10_eval_transform()

    # Two handles to the same underlying train data, different transforms.
    # Using Subset lets the val split get eval (no augmentation) transforms.
    dataset_train_aug = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform_train
    )
    dataset_train_eval = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform_eval
    )
    dataset_test = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform_eval
    )

    n_total = len(dataset_train_aug)  # 50 000
    n_val = int(n_total * val_split)  # 5 000
    n_train = n_total - n_val         # 45 000

    rng = torch.Generator().manual_seed(42)
    perm = torch.randperm(n_total, generator=rng)
    train_indices = perm[:n_train].tolist()
    val_indices = perm[n_train:].tolist()

    dataset_train = Subset(dataset_train_aug, train_indices)
    dataset_val = Subset(dataset_train_eval, val_indices)

    train_loader = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, test_loader, dataset_train, dataset_val, dataset_test


def get_cifar10_supcon_loaders(
    data_root: str,
    batch_size: int,
    num_workers: int = 0,
    smoke_test: bool = False,
    val_split: float = 0.1,
):
    """
    Returns CIFAR-10 loaders for supervised contrastive training and evaluation.

    The 50K training set is split into a contrastive-training subset and a
    validation subset (eval transforms, no augmentation).  The 10K test set
    is returned as a separate held-out loader.

    The training loader yields ``(views, labels)`` where ``views`` has shape
    ``(batch_size, 2, 3, 32, 32)``.

    Returns:
        train_loader, memory_loader, val_loader, test_loader,
        dataset_train, dataset_val, dataset_test
    """
    if smoke_test:
        n_train, n_val, n_test = 160, 40, 50
        train_images = torch.randn(n_train, 3, 32, 32)
        train_labels = torch.randint(0, 10, (n_train,))
        val_images = torch.randn(n_val, 3, 32, 32)
        val_labels = torch.randint(0, 10, (n_val,))
        test_images = torch.randn(n_test, 3, 32, 32)
        test_labels = torch.randint(0, 10, (n_test,))

        train_contrastive = TensorDataset(
            train_images.unsqueeze(1).repeat(1, 2, 1, 1, 1), train_labels
        )
        train_eval_ds = TensorDataset(train_images, train_labels)
        val_ds = TensorDataset(val_images, val_labels)
        test_ds = TensorDataset(test_images, test_labels)

        train_loader = DataLoader(train_contrastive, batch_size=batch_size, shuffle=True)
        memory_loader = DataLoader(train_eval_ds, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        return train_loader, memory_loader, val_loader, test_loader, train_contrastive, val_ds, test_ds

    contrastive_transform = get_cifar10_supcon_train_transform()
    eval_transform = get_cifar10_eval_transform()

    # Three handles to training data: contrastive aug, eval transforms (x2)
    train_contrastive_full = CIFAR10TwoViewDataset(
        root=data_root,
        train=True,
        download=True,
        base_transform=contrastive_transform,
    )
    train_eval_full = torchvision.datasets.CIFAR10(
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
    test_contrastive_dataset = CIFAR10DeterministicTwoViewDataset(
        root=data_root,
        train=False,
        download=True,
        base_transform=contrastive_transform,
        base_seed=0,
    )

    n_total = len(train_contrastive_full)  # 50 000
    n_val = int(n_total * val_split)       # 5 000
    n_train = n_total - n_val              # 45 000

    rng = torch.Generator().manual_seed(42)
    perm = torch.randperm(n_total, generator=rng)
    train_indices = perm[:n_train].tolist()
    val_indices = perm[n_train:].tolist()

    # Contrastive training uses train_indices with TwoCropTransform
    train_contrastive_dataset = Subset(train_contrastive_full, train_indices)
    # Memory loader (k-NN / embedding extraction) uses same indices, eval transforms
    train_memory_dataset = Subset(train_eval_full, train_indices)
    # Val uses disjoint indices, eval transforms
    val_dataset = Subset(train_eval_full, val_indices)

    train_loader = DataLoader(
        train_contrastive_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    memory_loader = DataLoader(
        train_memory_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return (
        train_loader,
        memory_loader,
        val_loader,
        test_loader,
        train_contrastive_dataset,
        val_dataset,
        test_eval_dataset,
    )


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


def get_cifar10_simclr_loaders(
    data_root: str,
    batch_size: int,
    num_workers: int = 0,
    smoke_test: bool = False,
    val_split: float = 0.1,
):
    """
    Returns CIFAR-10 loaders for self-supervised SimCLR pretraining.

    The training loader yields ``(views, labels)`` where ``views`` has shape
    ``(batch_size, 2, 3, 32, 32)``. Labels are not used during pretraining
    but are retained so the loader can be reused for kNN evaluation.

    A memory loader (eval transforms, no augmentation) is also returned for
    periodic kNN-accuracy monitoring.

    Returns:
        train_loader, memory_loader,
        train_dataset, memory_dataset
    """
    if smoke_test:
        n_train = 160
        train_images = torch.randn(n_train, 3, 32, 32)
        train_labels = torch.randint(0, 10, (n_train,))

        train_contrastive = TensorDataset(
            train_images.unsqueeze(1).repeat(1, 2, 1, 1, 1), train_labels
        )
        train_memory = TensorDataset(train_images, train_labels)

        train_loader = DataLoader(train_contrastive, batch_size=batch_size, shuffle=True)
        memory_loader = DataLoader(train_memory, batch_size=batch_size, shuffle=False)
        return train_loader, memory_loader, train_contrastive, train_memory

    simclr_transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
            p=0.8,
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
    ])
    eval_transform = get_cifar10_eval_transform()

    train_contrastive_full = torchvision.datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=TwoCropTransform(simclr_transform),
    )
    train_eval_full = torchvision.datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=eval_transform,
    )

    n_total = len(train_contrastive_full)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    rng = torch.Generator().manual_seed(42)
    perm = torch.randperm(n_total, generator=rng)
    train_indices = perm[:n_train].tolist()

    train_contrastive_dataset = Subset(train_contrastive_full, train_indices)
    train_memory_dataset = Subset(train_eval_full, train_indices)

    train_loader = DataLoader(
        train_contrastive_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    memory_loader = DataLoader(
        train_memory_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, memory_loader, train_contrastive_dataset, train_memory_dataset


_DINOV2_MEAN = (0.485, 0.456, 0.406)
_DINOV2_STD = (0.229, 0.224, 0.225)
_DINOV2_SIZE = 224


def get_cifar10_dinov2_loaders(
    data_root: str,
    batch_size: int,
    num_workers: int = 0,
    smoke_test: bool = False,
    val_split: float = 0.1,
):
    """
    Returns CIFAR-10 DataLoaders preprocessed for DINOv2.

    Images are resized from 32×32 to 224×224 (BICUBIC) and normalized with
    ImageNet statistics, matching the DINOv2 ViT pretraining distribution.
    Train/val split uses the same seed=42 permutation as get_cifar10_loaders.

    Returns:
        train_loader, val_loader, test_loader,
        dataset_train, dataset_val, dataset_test
    """
    if smoke_test:
        n_train, n_val, n_test = 160, 40, 50
        dataset_train = TensorDataset(
            torch.randn(n_train, 3, _DINOV2_SIZE, _DINOV2_SIZE),
            torch.randint(0, 10, (n_train,)),
        )
        dataset_val = TensorDataset(
            torch.randn(n_val, 3, _DINOV2_SIZE, _DINOV2_SIZE),
            torch.randint(0, 10, (n_val,)),
        )
        dataset_test = TensorDataset(
            torch.randn(n_test, 3, _DINOV2_SIZE, _DINOV2_SIZE),
            torch.randint(0, 10, (n_test,)),
        )
        train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader, dataset_train, dataset_val, dataset_test

    transform_train = transforms.Compose([
        transforms.Resize(_DINOV2_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(_DINOV2_MEAN, _DINOV2_STD),
    ])
    transform_eval = transforms.Compose([
        transforms.Resize(_DINOV2_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(_DINOV2_MEAN, _DINOV2_STD),
    ])

    dataset_train_aug = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform_train
    )
    dataset_train_eval = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform_eval
    )
    dataset_test = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform_eval
    )

    n_total = len(dataset_train_aug)
    n_val_ = int(n_total * val_split)
    n_train = n_total - n_val_

    rng = torch.Generator().manual_seed(42)
    perm = torch.randperm(n_total, generator=rng)
    train_indices = perm[:n_train].tolist()
    val_indices = perm[n_train:].tolist()

    dataset_train = Subset(dataset_train_aug, train_indices)
    dataset_val = Subset(dataset_train_eval, val_indices)

    train_loader = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        dataset_val, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        dataset_test, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader, dataset_train, dataset_val, dataset_test

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, TensorDataset, Sampler

import random
from collections import defaultdict

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

class MinPerClassSampler(Sampler):
    def __init__(self, labels, batch_size, min_per_class=2):
        self.labels = torch.tensor(labels)
        self.batch_size = batch_size
        self.min_per_class = min_per_class

        self.class_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.class_to_indices[int(label)].append(idx)

        self.classes = list(self.class_to_indices.keys())
        self.num_classes = len(self.classes)

    def __iter__(self):
        while True:
            batch = []

            # Step 1: ensure minimum per class (sample subset of classes)
            selected_classes = random.sample(
                self.classes,
                min(self.num_classes, self.batch_size // self.min_per_class)
            )

            for cls in selected_classes:
                indices = self.class_to_indices[cls]
                sampled = random.choices(indices, k=self.min_per_class)
                batch.extend(sampled)

            # Step 2: fill rest randomly
            remaining = self.batch_size - len(batch)
            if remaining > 0:
                all_indices = list(range(len(self.labels)))
                batch.extend(random.choices(all_indices, k=remaining))

            random.shuffle(batch)
            yield batch

    def __len__(self):
        return len(self.labels) // self.batch_size

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

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
    ])
    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
    ])

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
    no_aug: bool = False,
    min_per_class: int = 2,
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

        if no_aug:
            sampler = MinPerClassSampler(train_labels, batch_size=batch_size, min_per_class=min_per_class)
            train_loader = DataLoader(train_eval_ds, batch_size=batch_size, shuffle=True)
        else:
            train_loader = DataLoader(train_contrastive, batch_size=batch_size, shuffle=True)
        memory_loader = DataLoader(train_eval_ds, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        return train_loader, memory_loader, val_loader, test_loader, train_contrastive, val_ds, test_ds

    contrastive_transform = transforms.Compose([
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
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
    ])

    # Three handles to training data: contrastive aug, eval transforms (x2)
    train_transform = contrastive_transform if no_aug else TwoCropTransform(contrastive_transform)
    train_contrastive_full = torchvision.datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=train_transform,
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
    if no_aug:
        sampler = MinPerClassSampler(
            labels=torch.tensor(train_contrastive_dataset.dataset.targets
                                )[train_contrastive_dataset.indices],
            batch_size=batch_size,
            min_per_class=min_per_class,
        )
        train_loader = DataLoader(
            train_contrastive_dataset,
            num_workers=num_workers,
            pin_memory=True,
            batch_sampler=sampler,
        )
    else:
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

"""
CIFAR-10-C and CIFAR-100-C dataset loaders for corruption robustness evaluation.

Each corruption .npy file has shape (50000, 32, 32, 3) uint8, with the 5
severity levels concatenated (10000 images each). labels.npy has shape (50000,).

References:
  Hendrycks & Dietterich (2019) "Benchmarking Neural Network Robustness to
  Common Corruptions and Perturbations". ICLR 2019.
  CIFAR-10-C:  https://zenodo.org/record/2535967
  CIFAR-100-C: https://zenodo.org/record/3555552
"""

from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD = (0.2470, 0.2435, 0.2616)

_CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
_CIFAR100_STD = (0.2675, 0.2565, 0.2761)

CORRUPTIONS = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
]

_ZENODO_URLS = {
    "cifar10": "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar",
    "cifar100": "https://zenodo.org/record/3555552/files/CIFAR-100-C.tar",
}

_CIFARC_DIRS = {
    "cifar10": "CIFAR-10-C",
    "cifar100": "CIFAR-100-C",
}


class _DownloadProgressBar(tqdm):
    def update_to(self, b: int = 1, bsize: int = 1, tsize: int | None = None) -> None:
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def maybe_download_cifarc(data_root: str, dataset: str) -> str:
    """
    Ensures the CIFAR-C directory exists, downloading and extracting from
    Zenodo if necessary.

    Args:
        data_root: Parent directory where CIFAR-10-C/ or CIFAR-100-C/ should live.
        dataset:   "cifar10" or "cifar100".

    Returns:
        Absolute path to the CIFAR-C directory (e.g., /path/to/CIFAR-10-C/).
    """
    if dataset not in _ZENODO_URLS:
        raise ValueError(f"dataset must be 'cifar10' or 'cifar100', got '{dataset}'")

    root = Path(data_root)
    cifarc_dir = root / _CIFARC_DIRS[dataset]

    if cifarc_dir.exists():
        return str(cifarc_dir)

    url = _ZENODO_URLS[dataset]
    tar_path = root / f"{_CIFARC_DIRS[dataset]}.tar"

    print(f"{cifarc_dir} not found. Downloading from {url} ...")
    root.mkdir(parents=True, exist_ok=True)

    with _DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=tar_path.name) as pb:
        urllib.request.urlretrieve(url, tar_path, reporthook=pb.update_to)

    print(f"Extracting {tar_path} ...")
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=root)

    tar_path.unlink(missing_ok=True)
    print(f"Done. CIFAR-C data available at {cifarc_dir}")
    return str(cifarc_dir)


def get_cifarc_loader(
    cifarc_dir: str,
    corruption: str,
    severity: int,
    batch_size: int,
    num_workers: int = 0,
    id_normalization: str = "cifar10",
) -> tuple[DataLoader, torch.Tensor]:
    """
    Returns a DataLoader and label tensor for one corruption type and severity.

    The .npy file contains all 5 severities concatenated; this function slices
    the requested severity window and converts uint8 HWC -> float32 CHW,
    normalized with the ID dataset's mean/std.

    Args:
        cifarc_dir:       Path to the CIFAR-10-C/ or CIFAR-100-C/ directory.
        corruption:       One of the 15 corruption names in CORRUPTIONS.
        severity:         Integer in [1, 5].
        batch_size:       Mini-batch size.
        num_workers:      DataLoader workers.
        id_normalization: "cifar10" or "cifar100".

    Returns:
        (loader, labels) where labels is a (10000,) int64 tensor.
    """
    if corruption not in CORRUPTIONS:
        raise ValueError(f"Unknown corruption '{corruption}'. Must be one of: {CORRUPTIONS}")
    if severity not in range(1, 6):
        raise ValueError(f"severity must be in [1, 5], got {severity}")

    cifarc_path = Path(cifarc_dir)
    images_path = cifarc_path / f"{corruption}.npy"
    labels_path = cifarc_path / "labels.npy"

    if not images_path.exists():
        raise FileNotFoundError(
            f"Corruption file not found: {images_path}. "
            f"Run maybe_download_cifarc() first."
        )

    start = (severity - 1) * 10000
    end = severity * 10000

    # (10000, 32, 32, 3) uint8
    images_np = np.load(images_path)[start:end]
    labels_np = np.load(labels_path)[start:end]

    # (10000, 32, 32, 3) uint8 -> (10000, 3, 32, 32) float32 in [0, 1]
    images = torch.from_numpy(
        images_np.transpose(0, 3, 1, 2).astype(np.float32)
    ) / 255.0

    if id_normalization == "cifar100":
        mean = torch.tensor(_CIFAR100_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(_CIFAR100_STD, dtype=torch.float32).view(1, 3, 1, 1)
    else:
        mean = torch.tensor(_CIFAR10_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(_CIFAR10_STD, dtype=torch.float32).view(1, 3, 1, 1)

    images = (images - mean) / std

    labels = torch.from_numpy(labels_np.astype(np.int64))

    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return loader, labels

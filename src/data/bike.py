import os
import zipfile
import urllib.request
from math import floor

import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset


_FEATURE_COLS = [
    "season", "yr", "mnth", "hr", "holiday", "weekday",
    "workingday", "weathersit", "temp", "atemp", "hum", "windspeed",
]
_UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/"
    "Bike-Sharing-Dataset.zip"
)


def get_bike_loaders(
    data_path: str,
    batch_size: int,
    train_frac: float = 0.75,
    val_frac: float = 0.15,
    smoke_test: bool = False,
    device: str = "cpu",
):
    """
    Load, preprocess, and split the UCI Bike Sharing dataset into train/val/test.

    Fixes from the original notebook:
    - Creates train_loader and test_loader (DataLoader objects) that were missing.
    - Downloads the dataset automatically if hour.csv is absent.

    Args:
        data_path:  Path to hour.csv (downloaded here if missing).
        batch_size: Mini-batch size.
        train_frac: Fraction of data used for training (default 0.75).
        val_frac:   Fraction of data used for validation (default 0.15).
                    The remaining (1 - train_frac - val_frac) is used for test.
        smoke_test: If True, uses synthetic data (1000 samples, 3 features).
        device:     Device string to move tensors to.

    Returns:
        train_loader, val_loader, test_loader,
        train_x, train_y, val_x, val_y, test_x, test_y,
        train_n  (int, needed for DeepPredictiveLogLikelihood num_data)
    """
    if smoke_test:
        X = torch.randn(1000, 3)
        y = torch.randn(1000)
    else:
        if not os.path.isfile(data_path):
            zip_path = os.path.splitext(data_path)[0] + ".zip"
            print("Downloading UCI Bike Sharing dataset...")
            urllib.request.urlretrieve(_UCI_URL, zip_path)
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extract("hour.csv", path=os.path.dirname(data_path) or ".")

        df = pd.read_csv(data_path)
        data = torch.tensor(df[_FEATURE_COLS + ["cnt"]].values, dtype=torch.float32)

        X = data[:, :-1]
        X = X - X.min(0)[0]
        X = 2.0 * (X / X.max(0)[0]) - 1.0

        y = data[:, -1]
        y = (y - y.mean()) / y.std()

    shuffled = torch.randperm(X.size(0))
    X, y = X[shuffled], y[shuffled]

    n = X.size(0)
    train_n = int(floor(train_frac * n))
    val_n = int(floor(val_frac * n))

    train_x = X[:train_n].contiguous().to(device)
    train_y = y[:train_n].contiguous().to(device)
    val_x = X[train_n:train_n + val_n].contiguous().to(device)
    val_y = y[train_n:train_n + val_n].contiguous().to(device)
    test_x = X[train_n + val_n:].contiguous().to(device)
    test_y = y[train_n + val_n:].contiguous().to(device)

    train_loader = DataLoader(
        TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(val_x, val_y), batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(test_x, test_y), batch_size=batch_size, shuffle=False
    )

    return (
        train_loader, val_loader, test_loader,
        train_x, train_y, val_x, val_y, test_x, test_y,
        train_n,
    )

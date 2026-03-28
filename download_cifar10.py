#!/usr/bin/env python3
"""Download CIFAR-10 to a user-provided directory."""

import argparse
from pathlib import Path

import torchvision  # type: ignore[import-not-found]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download CIFAR-10 dataset to a target directory."
    )
    parser.add_argument(
        "data_root",
        type=Path,
        nargs="?",
        help="Directory where CIFAR-10 will be downloaded.",
        default=Path("/w/20252/wjcai/uq/datafolder"),
    )
    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument(
        "--train-only",
        action="store_true",
        help="Download only the training split.",
    )
    split_group.add_argument(
        "--test-only",
        action="store_true",
        help="Download only the test split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_root = args.data_root.expanduser().resolve()
    data_root.mkdir(parents=True, exist_ok=True)

    splits = []
    if args.train_only:
        splits = [("train", True)]
    elif args.test_only:
        splits = [("test", False)]
    else:
        splits = [("train", True), ("test", False)]

    for split_name, is_train in splits:
        torchvision.datasets.CIFAR10(root=data_root, train=is_train, download=True)
        print(f"Downloaded CIFAR-10 {split_name} split to: {data_root}")


if __name__ == "__main__":
    main()

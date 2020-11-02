from typing import Sequence

import torch
import torch.utils.data
import torchvision

from sklearn.metrics import accuracy_score

from src.path import DATA_PATH


def get_data(train: bool) -> torch.utils.data.Dataset:
    return torchvision.datasets.MNIST(
        DATA_PATH,
        train=train,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )


def get_d_in() -> int:
    # Input 28 x 28 pixels
    return 28 * 28


def get_d_out() -> int:
    # Digit classification from 0 to 9.
    return 10


def get_eval(pred: Sequence[int], ans: Sequence[int]) -> float:
    return accuracy_score(pred, ans)


def get_eval_name(train: bool) -> str:
    return 'train/acc' if train else 'test/acc'

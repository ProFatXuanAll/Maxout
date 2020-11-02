import torch
import torch.nn as nn

from src.model._base import BaseNN


class MaxoutNN(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_hid: int,
        d_out: int,
        k: int,
        p_in: int,
        p_hid: int
    ):
        super().__init__()
        self.l1 = [
            nn.Linear(in_features=d_in, out_features=d_hid)
            for _ in range(k)
        ]
        self.l2 = [
            nn.Linear(in_features=d_hid, out_features=d_hid)
            for _ in range(k)
        ]
        self.l3 = [
            nn.Linear(in_features=d_hid, out_features=d_hid)
            for _ in range(k)
        ]
        self.out = [
            nn.Linear(in_features=d_hid, out_features=d_out)
            for _ in range(k)
        ]

    def forward(self, x):
        h1 = torch.max(*[sub(x) for sub in self.l1])
        h2 = torch.max(*[sub(h1) for sub in self.l2])
        h3 = torch.max(*[sub(h2) for sub in self.l3])
        return torch.max(*[sub(h3) for sub in self.out])

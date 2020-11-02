import torch
import torch.nn as nn
import torch.nn.functional as F


class DropoutNN(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_hid: int,
        d_out: int,
        n_layer: int,
        p_in: int,
        p_hid: int
    ):
        super().__init__()

        layers = [
            nn.Dropout(p=p_in),
            nn.Linear(in_features=d_in, out_features=d_hid),
            nn.ReLU(),
            nn.Dropout(p=p_hid),
        ]

        for _ in range(n_layer - 1):
            layers.append(nn.Linear(in_features=d_hid, out_features=d_hid))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=p_hid))

        layers.append(nn.Linear(in_features=d_hid, out_features=d_out))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

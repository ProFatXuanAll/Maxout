import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model._base import BaseNN

class DropoutNN(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_hid: int,
        d_out: int,
        p_in: int,
        p_hid: int
    ):
        super().__init__()
        self.l1 = nn.Linear(in_features=d_in, out_features=d_hid)
        self.l2 = nn.Linear(in_features=d_hid, out_features=d_hid)
        self.l3 = nn.Linear(in_features=d_hid, out_features=d_hid)
        self.out = nn.Linear(in_features=d_hid, out_features=d_out)

        self.dp_in = nn.Dropout(p=p_in)
        self.dp_hid1 = nn.Dropout(p=p_hid)
        self.dp_hid2 = nn.Dropout(p=p_hid)
        self.dp_hid3 = nn.Dropout(p=p_hid)

    def forward(self, x):
        h1 = F.relu(self.l1(self.dp_in(x)))
        h2 = F.relu(self.l2(self.dp_hid1(h1)))
        h3 = F.relu(self.l3(self.dp_hid2(h2)))
        return self.out(self.dp_hid3(h3))

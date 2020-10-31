import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseNN(nn.Module):
    def __init__(
            self,
            d_in: int,
            d_hid: int,
            d_out: int
    ):
        super().__init__()
        self.l1 = nn.Linear(in_features=d_in, out_features=d_hid)
        self.l2 = nn.Linear(in_features=d_hid, out_features=d_hid)
        self.l3 = nn.Linear(in_features=d_hid, out_features=d_hid)
        self.out = nn.Linear(in_features=d_hid, out_features=d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        return self.out(h3)

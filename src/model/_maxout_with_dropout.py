import torch
import torch.nn as nn


class MaxoutWithDropoutNN(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_hid: int,
        d_out: int,
        k: int,
        n_layer: int,
        p_in: float,
        p_hid: float
    ):
        super().__init__()
        self.l_in = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(p=p_in),
                nn.Linear(in_features=d_in, out_features=d_hid)
            )
            for _ in range(k)
        ])
        self.l_hids = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Dropout(p=p_hid),
                    nn.Linear(in_features=d_hid, out_features=d_hid)
                )
                for _ in range(k)
            ])
            for _ in range(n_layer - 1)
        ])
        self.l_out = nn.Sequential(
            nn.Dropout(p=p_hid),
            nn.Linear(in_features=d_hid, out_features=d_out)
        )

    def forward(self, x):
        # (B, K, H)
        x = torch.stack([sub(x) for sub in self.l_in], dim=1)

        # (B, H)
        x = torch.max(x, dim=1)[0]

        for layer in self.l_hids:
            # (B, K, H)
            x = torch.stack([sub(x) for sub in layer], dim=1)

            # (B, H)
            x = torch.max(x, dim=1)[0]

        # (B, O)
        return self.l_out(x)

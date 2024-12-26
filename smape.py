import torch
import torch.nn as nn


class SMAPE(nn.Module):
    def __init__(self):
        super(SMAPE, self).__init__()

    def forward(self, pred, true):
        return torch.mean(
            200 * torch.abs(pred - true) / (torch.abs(pred) + torch.abs(true) + 1e-8)
        )

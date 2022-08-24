import torch


class Squeezer(torch.nn.Module):
    def forward(self, x):
        return x.squeeze(dim=-1)
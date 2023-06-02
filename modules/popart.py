import torch
from torch import nn as nn


class PopArt(nn.Module):
    def __init__(
        self,
        output_layer,
        beta: float = 0.0001,
        zero_debias: bool = False,
        start_pop: int = 0,
    ):
        # zero_debias=True and start_pop=8 seem to improve things a little but (False, 0) works as well
        super(PopArt, self).__init__()
        self.start_pop = start_pop
        self.zero_debias = zero_debias
        self.beta = beta
        self.output_layers = (
            output_layer
            if isinstance(output_layer, (tuple, list, nn.ModuleList))
            else (output_layer,)
        )
        # shape = self.output_layers[0].bias.shape
        shape = 1
        device = self.output_layers[0].bias.device
        # assert all(shape == x.bias.shape for x in self.output_layers)
        self.mean = nn.Parameter(
            torch.zeros(shape, device=device, dtype=torch.double), requires_grad=False
        )
        self.mean_square = nn.Parameter(
            torch.ones(shape, device=device, dtype=torch.double), requires_grad=False
        )
        self.std = nn.Parameter(
            torch.ones(shape, device=device, dtype=torch.double), requires_grad=False
        )
        self.updates = 0

    def forward(self, *input):
        pass

    @torch.no_grad()
    def update(self, targets):
        targets_shape = targets.shape
        targets = targets.view(-1, 1)
        beta = (
            max(1.0 / (self.updates + 1.0), self.beta)
            if self.zero_debias
            else self.beta
        )
        # note that for beta = 1/self.updates the resulting mean, std would be the true mean and std over all past data
        new_mean = (1.0 - beta) * self.mean + beta * targets.mean(0)
        new_mean_square = (1.0 - beta) * self.mean_square + beta * (
            targets * targets
        ).mean(0)
        new_std = (new_mean_square - new_mean *
                   new_mean).sqrt().clamp(0.0001, 1e6)
        assert self.std.shape == (1,), "this has only been tested in 1D"
        if self.updates >= self.start_pop:
            for layer in self.output_layers:
                layer.weight *= self.std / new_std
                layer.bias *= self.std
                layer.bias += self.mean - new_mean
                layer.bias /= new_std
        self.mean.copy_(new_mean)
        self.mean_square.copy_(new_mean_square)
        self.std.copy_(new_std)
        self.updates += 1
        return self.norm(targets).view(*targets_shape)

    def norm(self, x):
        return (x - self.mean) / self.std

    def unnorm(self, value):
        return value * self.std + self.mean

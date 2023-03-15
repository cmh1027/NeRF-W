import torch
import torch.nn as nn

class Gaussian(nn.Module):
    def __init__(self, sigma: float = 1.0):
        super(Gaussian, self).__init__()
        self.variance = sigma ** 2

    def forward(self, input):
        return torch.exp(-input**2 / (2 * self.variance))

class Sine(nn.Module):
    def __init__(self, freq: float = 1.0):
        super(Gaussian, self).__init__()
        self.freq = freq

    def forward(self, input):
        return torch.sin(2 * torch.pi * self.freq * input)


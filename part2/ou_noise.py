import torch
from copy import copy
from torch import nn


class OUNoise:
    """Ornstein-Uhlenbeck noise"""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):

        self.size = size
        self.mu = mu * torch.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = torch.manual_seed(seed)
        self.reset()

    def reset(self):
        self.state = copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * torch.normal(0, 1, self.size)
        self.state = x + dx
        return self.state
"""
    Models for direct methods
"""
import torch.nn as nn
import numpy as np



class LinearRegressionModel(nn.Module):
    """
        Linear regression model to regress the outcome
    """

    def __init__(self, input_dim, output_dim, n_hidden_layer, hidden_layer_size):
        super(LinearRegressionModel, self).__init__()
        self.model = nn.Sequential(nn.Linear(input_dim, output_dim))

    def forward(self, x):
        out = self.model(x)
        return out


class MLP(nn.Module):
    """
        MLP as direct methods to regress the outcome
    """

    def __init__(self, input_dim, output_dim, n_hidden_layer, hidden_layer_size):
        super(MLP, self).__init__()
        self.model = nn.Sequential(nn.Linear(input_dim, hidden_layer_size),
                                   nn.LeakyReLU(),
                                   nn.Linear(hidden_layer_size, hidden_layer_size),
                                   nn.LeakyReLU(),
                                   nn.Linear(hidden_layer_size, hidden_layer_size),
                                   nn.LeakyReLU(),
                                   nn.Linear(hidden_layer_size, hidden_layer_size),
                                   nn.LeakyReLU(),
                                   nn.Linear(hidden_layer_size, 1)
                                   )

    def forward(self, x):
        out = self.model(x)
        return out

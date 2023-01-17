import numpy as np
import torch
from torch import nn, optim
from torch.nn.modules import Module
from torch.optim import Optimizer


class MLP(Module):
    """
    A general MLP class. Initialisation example:
    mlp = MLP(input_size, 64, ReLU(), 64, ReLU(), output_size, Sigmoid())
    """

    default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, input_size, *layers_data, device=default_device, learning_rate=0.01,
                 optimizer_class=optim.Adam):
        """
        For each element in layers_data:
         - If the element is an integer, it will be replaced by a linear layer with this integer as output size,
         - If this is a model (like activation layer) il will be directly integrated
         - If it is a function, it will be used to initialise the weights of the layer before
            So we call layer_data[n](layer_data[n - 1].weights) with n the index of the activation function in
            layers_data
        """
        super().__init__()
        assert issubclass(optimizer_class, Optimizer)

        self.layers = nn.ModuleList()
        self.input_size = input_size  # Can be useful later ...
        for data in layers_data:
            layer = data
            if isinstance(data, int):
                layer = nn.Linear(input_size, data)
                input_size = data
            if callable(data) and not isinstance(data, nn.Module):
                data(self.layers[-1].weight)
                continue
            self.layers.append(layer)  # For the next layer

        self.device = device
        self.to(self.device)
        self.learning_rate = learning_rate
        self.optimizer = optimizer_class(params=self.parameters(), lr=learning_rate)

    def forward(self, input_data):
        if isinstance(input_data, np.ndarray):
            input_data = torch.from_numpy(input_data)
        input_data = input_data.float()
        for layer in self.layers:
            input_data = layer(input_data)
        return input_data

    def converge_to(self, other_mlp, tau=0.1):
        for self_param, other_param in zip(self.parameters(), other_mlp.get_parameters()):
            self_param.data.copy_(
                self_param.data * (1.0 - tau) + other_param.data * tau
            )

    def learn(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

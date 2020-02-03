from typing import Union
import numpy as np
import scipy.signal
import torch

def combine_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def gae(gamma, lambda_, rewards: Union[np.ndarray, torch.Tensor], predicted_values: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    deltas = rewards[:-1] + gamma * predicted_values[1:] - predicted_values[:-1]
    return discount_cumsum(deltas, gamma * lambda_)



class MlpModel(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size,
                 activation: torch.nn.Module = torch.nn.Tanh(),
                 output_activation: torch.nn.Module = None,
                 device=torch.device('cpu')):
        super(MlpModel, self).__init__()

        self.activation: torch.nn.Module = activation
        self.output_activation: torch.nn.Module = output_activation
        self.layers = []
        previous_size = input_size
        for h in hidden_sizes:
            layer = torch.nn.Linear(previous_size, h).to(device)
            # layer.weight.data.fill_(0.1)
            # layer.bias.data.fill_(0.1)
            self.layers.append(layer)
            previous_size = h
        self.final_layer = torch.nn.Linear(previous_size, output_size).to(device)
        # self.final_layer.weight.data.fill_(0.1)
        # self.final_layer.bias.data.fill_(0.1)

    def forward(self, x):
        for layer in self.layers:
            val = layer(x)
            x = self.activation(val)
        output = self.final_layer(x)
        if self.output_activation is not None:
            output = self.output_activation(output)
        return output

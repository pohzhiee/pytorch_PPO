from typing import Union
import numpy as np
import scipy.signal
import torch
from torch.nn import Sequential, Linear

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


class MlpModel(Sequential):
    def __init__(self, input_size, hidden_sizes, output_size, activation: torch.nn.Module = torch.nn.Tanh(),
                 output_activation: torch.nn.Module = None, device=torch.device('cpu')):
        super().__init__()
        previous_size = input_size
        for i, h in enumerate(hidden_sizes):
            self.add_module(f'layer{i+1}', Linear(previous_size, h).to(device))
            self.add_module(f'layer{i+1}_activation', activation)
            previous_size = h
        self.add_module(f'output_layer', Linear(previous_size, output_size).to(device))
        if output_activation is not None:
            self.add_module(f'output_activation', output_activation)

import torch
import numpy as np


class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size2, output_size):
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.tanh = torch.nn.Tanh()
        self.fc2 = torch.nn.Linear(self.hidden_size, hidden_size2)
        self.fc3 = torch.nn.Linear(self.hidden_size2, output_size)

        # self.fc1.weight.data.fill_(0.1)
        # self.fc1.bias.data.fill_(0.1)
        # self.fc2.weight.data.fill_(0.1)
        # self.fc2.bias.data.fill_(0.1)
        # self.fc3.weight.data.fill_(0.1)
        # self.fc3.bias.data.fill_(0.1)

    def forward(self, x):
        hidden = self.fc1(x)
        tanh = self.tanh(hidden)
        output1 = self.fc2(tanh)
        tanh2 = self.tanh(output1)
        output = self.fc3(tanh2)
        return output


class Model2(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size,
                 activation: torch.nn.Module = torch.nn.Tanh(),
                 output_activation: torch.nn.Module = None):
        super(Model2, self).__init__()
        self.activation: torch.nn.Module = activation
        self.output_activation: torch.nn.Module = output_activation
        self.layers = []
        previous_size = input_size
        for h in hidden_sizes:
            layer = torch.nn.Linear(previous_size, h)
            # layer.weight.data.fill_(0.1)
            # layer.bias.data.fill_(0.1)
            self.layers.append(layer)
            previous_size = h
        self.final_layer = torch.nn.Linear(previous_size, output_size)
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


v_model = Model(4, 64, 64, 1)
obs_tensor = torch.from_numpy(np.array([[1.0, 1.0, 1.0, 1.0], [1.0,1.0,1.0,1.0]])).float()
v_model2 = Model2(4, (64, 64), 1, torch.nn.Tanh())
v = v_model(obs_tensor)
v2 = v_model2(obs_tensor)
print(v)
print(v2)
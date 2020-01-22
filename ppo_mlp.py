import torch


class MlpModel(torch.nn.Module):
    def __init__(self, input_size: int, hidden_sizes, output_size: int,
                 activation: torch.nn.Module = torch.nn.Tanh(),
                 output_activation: torch.nn.Module = None):
        super(MlpModel, self).__init__()

        self.activation: torch.nn.Module = activation
        self.output_activation: torch.nn.Module = output_activation
        self.layers = []
        previous_size = input_size
        for h in hidden_sizes:
            layer = torch.nn.Linear(previous_size, h)
            layer.weight.data.fill_(0.1)
            layer.bias.data.fill_(0.1)
            self.layers.append(layer)
            previous_size = h
        self.final_layer = torch.nn.Linear(previous_size, output_size)
        self.final_layer.weight.data.fill_(0.1)
        self.final_layer.bias.data.fill_(0.1)

    def forward(self, x):
        for layer in self.layers:
            val = layer(x)
            x = self.activation(val)
        output = self.final_layer(x)
        if self.output_activation is not None:
            output = self.output_activation(output)
        return output

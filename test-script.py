import torch

x = torch.randn(8, 3)
indices = torch.tensor([0, 1, 1, 0, 1, 1, 0, 1])
indices_view = indices.view(-1, 1)
y = x.gather(1, indices_view)
print(x)
print(indices)
print(indices_view)
print(y)

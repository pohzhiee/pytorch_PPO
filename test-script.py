# import torch
#
# x = torch.randn(8, 3)
# indices = torch.tensor([0, 1, 1, 0, 1, 1, 0, 1])
# indices_view = indices.view(-1, 1)
# y = x.gather(1, indices_view)
# print(x)
# print(indices)
# print(indices_view)
# print(y)
import numpy

a = numpy.array([1, 2, 3, 4, 5, 6.0])
b = numpy.array([-1, -2, -3, -4, -5, -6, -7.0])

for _ in range(1000):
    c = numpy.concatenate([a, b])

for _ in range(1000):
    d =

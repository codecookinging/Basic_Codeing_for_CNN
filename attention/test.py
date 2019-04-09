import torch
import torch.nn as nn

m= nn.AdaptiveAvgPool2d((99,99))

input = torch.randn((1,64,8,8))
print(m(input).shape)

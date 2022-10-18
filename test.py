import torch
a = torch.randn(4, 3)
print(a)
print(a.argmax(1) + 1)
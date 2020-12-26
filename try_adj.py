import torch


x = torch.tensor([[1,2,3,4],[5,6,7,8]])
y = torch.unsqueeze(x, 1)
y = y.expand(2,3,4)
print(x.shape)
print(y.shape)
print(y)
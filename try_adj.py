import pickle
import pandas as pd
import numpy as np
import torch

# with open('/home/wenjing/Desktop/onet_2d_64/pretrained/time_generation.pkl', 'rb') as f:
#     data = pickle.load(f)
#
# print(data)

# d = pd.read_pickle('/home/wenjing/Desktop/onet/pretrained/time_generation.pkl')
# d.to_csv('/home/wenjing/Desktop/onet/pretrained/time_generation.csv')
W = 13
H = 13
i, j = torch.meshgrid(torch.linspace(0, W - 1, W),
                              torch.linspace(0, H - 1, H))  # pytorch's meshgrid has indexing='ij'
i = i.t() #transpose
j = j.t()
print(i)
print('##', j)
i = i.reshape(-1, 1)
j = j.reshape(-1, 1)
print(')', i)
print('!!!!!', j)

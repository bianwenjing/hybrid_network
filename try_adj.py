import pickle
import pandas as pd
import numpy as np

# with open('/home/wenjing/Desktop/onet_2d_64/pretrained/time_generation.pkl', 'rb') as f:
#     data = pickle.load(f)
#
# print(data)

# d = pd.read_pickle('/home/wenjing/Desktop/onet/pretrained/time_generation.pkl')
# d.to_csv('/home/wenjing/Desktop/onet/pretrained/time_generation.csv')
a = np.ones((3,4,5))
index = [1,2]
index2=[0,1,2]
print(a[:, index2,:][:,:,index].shape)
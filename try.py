import numpy as np
import pickle

# data = np.load('/home/wenjing/Downloads/ShapeNet/03211117/717d3ef4c7000c5917a27c3fa1d3566d/img_choy2016/cameras.npz')
# lst = data.files
# for item in lst:
#     print(item)
#     print(data[item])
# base_mesh = np.loadtxt('im2mesh/pix2mesh/ellipsoid/face1.obj', dtype='|S32')
# print(base_mesh)

# ellipsoid = pickle.load(open('im2mesh/pix2mesh/ellipsoid/info_ellipsoid.dat', 'rb'), encoding='latin1')
# print(ellipsoid[1][1][0])
# # ellipsoid[0] initial coordinates
# # ellipsoid[1][1] ellipsoid[2][1] ellipsoid[3][1] support_array (tnsor): sparse weighted adjencency matrix
# #                 with non-zero entries on the diagonal
# # ellipsoid[4][0] ellipsoid[4][1] # IDs for the first/second unpooling operation,pool_idx_array (tensor): vertex IDs that should be combined to new
# #             vertices
#
#
# import os
# path = "/media/wenjing/Data2/ShapeNet.build/03001627/2_watertight/"
# filelist=os.listdir(path)
# for item in filelist:
#     if 'off' not in item:
#         print(item)
import torch
# from torch import distributions as dist
# #
# # a = torch.tensor([5, 6])
# # p_r = dist.Bernoulli(logits=a)
# # print(p_r.probs)
#
# def removekey(d, listofkeys):
#     r = dict(d)
#     for key in listofkeys:
#         print('key: {} is removed'.format(key))
#         r.pop(key)
#     return r
#
#
#
model_dict = torch.load('/home/wenjing/storage/conv2d/2code/model.pt')
# print(model_dict)
# list = []
for key in model_dict['model']:
    print(key)
#     if 'init' in key:
#         list.append(key)
# new_model = {}
# new_model['model'] = removekey(model_dict['model'], list)
# torch.save(new_model, '/home/wenjing/Desktop/no_init.pt')
#
# model_dict = torch.load('/home/wenjing/Desktop/no_init.pt')
# # print(model_dict.keys())
# for key in model_dict['model']:
# #     print(key, model_dict['model'][key].shape)
# from torch.utils import model_zoo
#
# url = 'https://s3.eu-central-1.amazonaws.com/avg-projects/occupancy_networks/models/onet_img2mesh_3-f786b04a.pt'
# # url = 'https://s3.eu-central-1.amazonaws.com/avg-projects/occupancy_networks/models/onet_uncond_airplanes-26c9d089.pt'
# model2 = model_zoo.load_url(url, progress=True)
# for key in model2['model']:
#     print(key, model2['model'][key].shape)
# list = []
# for key in model2['model']:
#     if 'encoder' not in key:
#         list.append(key)
#     print(key, model2['model'][key].shape)
# new_model = {}
# new_model['model'] = removekey(model2['model'], list)
# torch.save(new_model, '/home/wenjing/uncon_encoder.pt')
# model_dict = torch.load('/home/wenjing/storage/psgn_encoder/model.pt')
# for key in model_dict['model']:
#     print(key)
# print(model_dict['model']['encoder.fc.weight'])
# for key in model_dict['model']:
#     if 'init_pts' in key:
#         print(key, model_dict['model'][key].shape, model_dict['model'][key])
###################dict convert##########################################

# model2 = torch.load('/home/wenjing/Desktop/model_800000.pt')
# list2 = []
# for key in model2['model']:
#     if 'encoder' in key:
#         list2.append(key)
# # print(len(list2))
# model_dict = torch.load('/home/wenjing/Downloads/tensorflow.pth.tar')
# list = []
# for key in model_dict['model']:
#     if 'encoder' in key:
#         list.append(key)
# # print(len(list))
# dic = {}
# for i in range(len(list)):
#     dic[list[i]] = list2[i]
#
# new_model={}
# new_model['model'] = {}
# model3 = torch.load('/home/wenjing/Desktop/encoder.pt')
# for key in model3['model']:
#     key_oc = dic[key]
#     new_model['model'][key_oc] = model3['model'][key]
#
# torch.save(new_model, '/home/wenjing/occupancy_networks-master/out/img/pixel2mesh/encoder.pt')
#######################################################################################
# model1 = torch.load('/home/wenjing/occupancy_networks-master/out/img/pixel2mesh/model_0.pt')
# print(model1["model"]["decoder.gc1.0.lin.bias"])
#
# model2 = torch.load('/home/wenjing/occupancy_networks-master/out/img/pixel2mesh/model.pt')
# print(model2["model"]['decoder.gc1.0.lin.bias'])
import torch

def removekey(d, listofkeys):
    r = dict(d)
    for key in listofkeys:
        print('key: {} is removed'.format(key))
        r.pop(key)
    return r



model_dict = torch.load('/home/wenjing/storage/model_best.pt')
# model_dict = torch.load('/home/wenjing/occupancy_networks-master/out/img/pixel2mesh/model.pt')
list = []
for key in model_dict['model']:
    if 'adj_mat' in key or 'init_pts' in key:
        list.append(key)
new_model = {}
new_model['model'] = removekey(model_dict['model'], list)

# torch.save(new_model, '/home/wenjing/occupancy_networks-master/out/img/pixel2mesh/model_remove.pt')
torch.save(new_model, '/home/wenjing/storage/model_best.pt')
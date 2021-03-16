import yaml
from torchvision import transforms
from im2mesh import data
from im2mesh import onet, r2n2, psgn, pix2mesh, dmc, combined, combined2, onet_2d, onet_2d_conv, \
    onet_2d_2code, onet_1d, onet_2d_shared, onet_2d_cam, onet_2d_pointmap, onet_2d_depth
from im2mesh import preprocess
from multiprocessing import Manager

method_dict = {
    'onet': onet,
    'r2n2': r2n2,
    'psgn': psgn,
    'pix2mesh': pix2mesh,
    'dmc': dmc,
    'combined': combined,
    'combined2': combined2,
    'onet_2d': onet_2d,
    'conv_onet_2d': onet_2d_conv,
    'conv_2code': onet_2d_2code,
    'onet_1d': onet_1d,
    'shared_2code': onet_2d_shared,
    'onet_2d_cam': onet_2d_cam,
    'onet_2d_pointmap': onet_2d_pointmap,
    'onet_2d_depth': onet_2d_depth,
}


# General config
def load_config(path, default_path=None):
    ''' Loads config file.

    Args:  
        path (str): path to config file
        default_path (bool): whether to use default path
    '''
    # Load configuration from file itself
    with open(path, 'r') as f:
        cfg_special = yaml.load(f)

    # Check if we should inherit from a config
    inherit_from = cfg_special.get('inherit_from')

    # If yes, load this config first as default
    # If no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.load(f)
    else:
        cfg = dict()

    # Include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    ''' Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used

    '''
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


# Models
def get_model(cfg, device=None, dataset=None):
    ''' Returns the model instance.

    Args:
        cfg (dict): config dictionary
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    method = cfg['method']
    model = method_dict[method].config.get_model(
        cfg, device=device, dataset=dataset)
    return model

########################################################
def get_model2(cfg, device=None, dataset=None):
    ''' Returns the model instance.

    Args:
        cfg (dict): config dictionary
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    method1 = cfg['method1']
    method2 = cfg['method2']
    # model = method_dict[method].config.get_model(
    #     cfg, device=device, dataset=dataset)
    model1 = method_dict[method1].config.get_model(
        cfg, device=device, dataset=dataset)
    model2 = method_dict[method2].config.get_model(
        cfg, device=device, dataset=dataset)
    return model1, model2
#########################################################
# Trainer
def get_trainer(model, optimizer, cfg, device):
    ''' Returns a trainer instance.

    Args:
        model (nn.Module): the model which is used
        optimizer (optimizer): pytorch optimizer
        cfg (dict): config dictionary
        device (device): pytorch device
    '''
    method = cfg['method']
    trainer = method_dict[method].config.get_trainer(
        model, optimizer, cfg, device)
    return trainer


# Generator for final mesh extraction
def get_generator(model, cfg, device):
    ''' Returns a generator instance.

    Args:
        model (nn.Module): the model which is used
        cfg (dict): config dictionary
        device (device): pytorch device
    '''
    method = cfg['method']
    generator = method_dict[method].config.get_generator(model, cfg, device)
    return generator

def get_generator2(model1, model2, cfg, device):
    ''' Returns a generator instance.

    Args:
        model (nn.Module): the model which is used
        cfg (dict): config dictionary
        device (device): pytorch device
    '''
    method = cfg['method']
    generator = method_dict[method].config.get_generator(model1, model2, cfg, device)
    return generator

# Datasets
def get_dataset(mode, cfg, return_idx=False, return_category=False):
    ''' Returns the dataset.

    Args:
        model (nn.Module): the model which is used
        cfg (dict): config dictionary
        return_idx (bool): whether to include an ID field
    '''
    method = cfg['method']
    dataset_type = cfg['data']['dataset']
    dataset_folder = cfg['data']['path']
    categories = cfg['data']['classes']

    # Get split
    splits = {
        'train': cfg['data']['train_split'],
        'val': cfg['data']['val_split'],
        'test': cfg['data']['test_split'],
    }

    split = splits[mode]

    # Create dataset
    if dataset_type == 'Shapes3D':
        # Dataset fields
        # Method specific fields (usually correspond to output)
        fields = method_dict[method].config.get_data_fields(mode, cfg)
        # Input fields
        inputs_field = get_inputs_field(mode, cfg)
        if inputs_field is not None:
            fields['inputs'] = inputs_field

        if return_idx:
            fields['idx'] = data.IndexField()

        if return_category:
            fields['category'] = data.CategoryField()

        dataset = data.Shapes3dDataset(
            dataset_folder, fields,
            split=split,
            categories=categories,
        )
    elif dataset_type == 'kitti':
        dataset = data.KittiDataset(
            dataset_folder, img_size=cfg['data']['img_size'],
            return_idx=return_idx
        )
    elif dataset_type == 'online_products':
        dataset = data.OnlineProductDataset(
            dataset_folder, img_size=cfg['data']['img_size'],
            classes=cfg['data']['classes'],
            max_number_imgs=cfg['generation']['max_number_imgs'],
            return_idx=return_idx, return_category=return_category
        )
    elif dataset_type == 'images':
        dataset = data.ImageDataset(
            dataset_folder, img_size=cfg['data']['img_size'],
            return_idx=return_idx,
        )
    else:
        raise ValueError('Invalid dataset "%s"' % cfg['data']['dataset'])
 
    return dataset

def get_dataset_depth(mode ,cfg, return_idx=False, return_category=False,
                **kwargs):
    ''' Returns a dataset instance.

    Args:
        cfg (dict): config dictionary
        mode (string): which mode is used (train / val /test / render)
        return_idx (bool): whether to return model index
        return_category (bool): whether to return model category
    '''
    # Get fields with cfg
    method = cfg['method']
    input_type = cfg['data']['input_type']
    dataset_name = cfg['data']['dataset']
    dataset_folder = cfg['data']['path']

    categories = cfg['data']['classes']
    cache_fields = cfg['data']['cache_fields']
    n_views = cfg['data']['n_views']
    split_model_for_images = cfg['data']['split_model_for_images']

    splits = {
        'train': cfg['data']['train_split'],
        'val': cfg['data']['val_split'],
        'test': cfg['data']['test_split'],
        'render': cfg['data']['test_split'],
    }
    split = splits[mode]
    fields = method_dict[method].config.get_data_fields(mode, cfg)

    if input_type == 'idx':
        input_field = data.IndexField()
        fields['inputs'] = input_field
    elif input_type == 'img':
        random_view = True if \
            (mode == 'train' or dataset_name == 'NMR') else False
        resize_img_transform = data.ResizeImage(cfg['data']['img_size_input'])
        fields['inputs'] = data.ImagesField_depth(
            cfg['data']['img_folder_input'],
            transform=resize_img_transform,
            with_mask=False, with_camera=False,
            extension=cfg['data']['img_extension_input'],
            n_views=cfg['data']['n_views_input'], random_view=random_view)

    else:
        input_field = None

    if return_idx:
        fields['idx'] = data.IndexField()

    if return_category:
        fields['category'] = data.CategoryField()

    manager = Manager()
    shared_dict = manager.dict()

    if ((dataset_name == 'Shapes3D') or
        (dataset_name == 'DTU') or
            (dataset_name == 'NMR')):
        dataset = data.Shapes3dDataset_depth(
            dataset_folder, fields, split=split,
            categories=categories,
            shared_dict=shared_dict,
            n_views=n_views, cache_fields=cache_fields,
            split_model_for_images=split_model_for_images)
    elif dataset_name == 'images':
        dataset = data.ImageDataset(
            dataset_folder, return_idx=True
        )
    else:
        raise ValueError('Invalid dataset_name!')

    return dataset

def get_inputs_field(mode, cfg):
    ''' Returns the inputs fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): config dictionary
    '''
    input_type = cfg['data']['input_type']
    with_transforms = cfg['data']['with_transforms']

    if input_type is None:
        inputs_field = None
    elif input_type == 'img':
        if mode == 'train' and cfg['data']['img_augment']:
            resize_op = transforms.RandomResizedCrop(
                cfg['data']['img_size'], (0.75, 1.), (1., 1.))
        else:
            resize_op = transforms.Resize((cfg['data']['img_size']))

        transform = transforms.Compose([
            resize_op, transforms.ToTensor(),
        ])

        with_camera = cfg['data']['img_with_camera']

        if mode == 'train':
            random_view = True
        else:
            random_view = False

        inputs_field = data.ImagesField(
            cfg['data']['img_folder'], transform,
            with_camera=with_camera, random_view=random_view
        )
    elif input_type == 'pointcloud':
        transform = transforms.Compose([
            data.SubsamplePointcloud(cfg['data']['pointcloud_n']),
            data.PointcloudNoise(cfg['data']['pointcloud_noise'])
        ])
        with_transforms = cfg['data']['with_transforms']
        inputs_field = data.PointCloudField(
            cfg['data']['pointcloud_file'], transform,
            with_transforms=with_transforms
        )
    elif input_type == 'voxels':
        inputs_field = data.VoxelsField(
            cfg['data']['voxels_file']
        )
    elif input_type == 'idx':
        inputs_field = data.IndexField()
    else:
        raise ValueError(
            'Invalid input type (%s)' % input_type)
    return inputs_field


def get_preprocessor(cfg, dataset=None, device=None):
    ''' Returns preprocessor instance.

    Args:
        cfg (dict): config dictionary
        dataset (dataset): dataset
        device (device): pytorch device
    '''
    p_type = cfg['preprocessor']['type']
    cfg_path = cfg['preprocessor']['config']
    model_file = cfg['preprocessor']['model_file']

    if p_type == 'psgn':
        preprocessor = preprocess.PSGNPreprocessor(
            cfg_path=cfg_path,
            pointcloud_n=cfg['data']['pointcloud_n'],
            dataset=dataset,
            device=device,
            model_file=model_file,
        )
    elif p_type is None:
        preprocessor = None
    else:
        raise ValueError('Invalid Preprocessor %s' % p_type)

    return preprocessor

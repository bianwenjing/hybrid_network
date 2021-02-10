import torch
import torch.distributions as dist
from torch import nn
import os
from im2mesh.encoder import encoder_dict
from im2mesh.onet_2d_2code import models, training, generation
from im2mesh import data
from im2mesh import config


def get_model(cfg, device=None, dataset=None, **kwargs):
    ''' Return the Occupancy Network model.

    Args:
        cfg (dict): imported yaml config
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    decoder = cfg['model']['decoder']
    encoder_local = cfg['model']['encoder_local']
    encoder_global = cfg['model']['encoder_global']
    # encoder_latent = cfg['model']['encoder_latent']
    dim = cfg['data']['dim']
    # z_dim = cfg['model']['z_dim']
    c_dim_local = cfg['model']['c_dim_local']
    c_dim_global = cfg['model']['c_dim_global']
    decoder_kwargs = cfg['model']['decoder_kwargs']
    encoder_kwargs = cfg['model']['encoder_kwargs']
    # encoder_latent_kwargs = cfg['model']['encoder_latent_kwargs']
    z_resolution = cfg['model']['z_resolution']
    padding = cfg['data']['padding']

    decoder = models.decoder_dict[decoder](z_resolution=z_resolution,
        dim=dim, c_dim_local=c_dim_local,c_dim_global=c_dim_global, padding=padding,
        **decoder_kwargs
    )

    # if z_dim != 0:
    #     encoder_latent = models.encoder_latent_dict[encoder_latent](
    #         z_resolution=z_resolution,
    #         dim=dim, z_dim=z_dim, c_dim=c_dim,
    #         **encoder_latent_kwargs
    #     )
    # else:
    #     encoder_latent = None

    if encoder_local == 'idx':
        encoder_local = nn.Embedding(len(dataset), c_dim_local)
    elif encoder_local is not None:
        encoder_local = encoder_dict[encoder_local](
            c_dim=c_dim_local,
            **encoder_kwargs
        )
    else:
        encoder_local = None

    if encoder_global == 'idx':
        encoder_global = nn.Embedding(len(dataset), c_dim_global)
    elif encoder_global is not None:
        encoder_global = encoder_dict[encoder_global](
            c_dim=c_dim_global,
            **encoder_kwargs
        )
    else:
        encoder_global = None

    model = models.OccupancyNetwork(
        decoder, encoder_local, encoder_global, device=device
    )

    return model


def get_trainer(model, optimizer, cfg, device, **kwargs):
    ''' Returns the trainer object.

    Args:
        model (nn.Module): the Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    threshold = cfg['test']['threshold']
    out_dir = cfg['training']['out_dir']
    vis_dir = os.path.join(out_dir, 'vis')
    input_type = cfg['data']['input_type']
    z_resolution = cfg['model']['z_resolution']
    camera = cfg['data']['img_with_camera']

    trainer = training.Trainer(
        model, optimizer,
        device=device, input_type=input_type,
        vis_dir=vis_dir, threshold=threshold,
        eval_sample=cfg['training']['eval_sample'],
        z_resolution=z_resolution,
        camera=camera
    )

    return trainer


def get_generator(model, cfg, device, **kwargs):
    ''' Returns the generator object.

    Args:
        model (nn.Module): Occupancy Network model
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    preprocessor = config.get_preprocessor(cfg, device=device)

    generator = generation.Generator3D(
        model,
        device=device,
        threshold=cfg['test']['threshold'],
        z_resolution=cfg['model']['z_resolution'],
        resolution0=cfg['generation']['resolution_0'],
        upsampling_steps=cfg['generation']['upsampling_steps'],
        sample=cfg['generation']['use_sampling'],
        refinement_step=cfg['generation']['refinement_step'],
        simplify_nfaces=cfg['generation']['simplify_nfaces'],
        preprocessor=preprocessor,
    )
    return generator


def get_prior_z(cfg, device, **kwargs):
    ''' Returns prior distribution for latent code z.

    Args:
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    z_dim = cfg['model']['z_dim']
    p0_z = dist.Normal(
        torch.zeros(z_dim, device=device),
        torch.ones(z_dim, device=device)
    )

    return p0_z


def get_data_fields(mode, cfg):
    ''' Returns the data fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): imported yaml config
    '''
    points_transform = data.SubsamplePoints(cfg['data']['points_subsample'])
    # with_transforms = cfg['model']['use_camera']
    with_transforms = cfg['data']['with_transforms']
    z_resolution = cfg['model']['z_resolution']
    fields = {}
    fields['points'] = data.RayField2(
        cfg['data']['points_file'], points_transform,
        z_resolution=z_resolution,
        with_transforms=with_transforms,
        unpackbits=cfg['data']['points_unpackbits'],
    )

    if mode in ('val', 'test'):
        points_iou_file = cfg['data']['points_iou_file']
        voxels_file = cfg['data']['voxels_file']
        voxels_file = None
        if points_iou_file is not None:
            fields['points_iou'] = data.RayField2(
                points_iou_file,
                z_resolution=z_resolution,
                with_transforms=with_transforms,
                unpackbits=cfg['data']['points_unpackbits'],
            )
        if voxels_file is not None:
            fields['voxels'] = data.VoxelsField(voxels_file)

    # fields = {}
    # fields['points'] = data.RayField(
    #     cfg['data']['voxels_file'], points_transform,
    #     with_transforms=with_transforms,
    # )
    #
    # if mode in ('val', 'test'):
    #     points_iou_file = cfg['data']['voxels_file']
    #     voxels_file = cfg['data']['voxels_file']
    #     if points_iou_file is not None:
    #         fields['points_iou'] = data.RayField(
    #             points_iou_file,
    #             with_transforms=with_transforms,
    #         )
    #     if voxels_file is not None:
    #         fields['voxels'] = data.VoxelsField(voxels_file)


    return fields

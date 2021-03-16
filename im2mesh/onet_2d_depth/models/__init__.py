import torch
import torch.nn as nn
from torch import distributions as dist
from im2mesh.onet_2d_depth.models import decoder

# # Encoder latent dictionary
# encoder_latent_dict = {
#     'simple': encoder_latent.Encoder,
# }

# # Decoder dictionary
# decoder_dict = {
#     'simple': decoder.Decoder,
#     'cbatchnorm': decoder.DecoderCBatchNorm,
#     'cbatchnorm2': decoder.DecoderCBatchNorm2,
#     'batchnorm': decoder.DecoderBatchNorm,
#     'cbatchnorm_noresnet': decoder.DecoderCBatchNormNoResnet,
#
# }
# Decoder dictionary
decoder_dict = {
    'simple_local': decoder.LocalDecoder,
    'cbatchnorm_local': decoder.LocalDecoder_CBatchNorm,
    'simple_local_crop': decoder.PatchLocalDecoder,
    'simple_local_point': decoder.LocalPointDecoder
}



class OccupancyNetwork(nn.Module):
    ''' Occupancy Network class.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        encoder_latent (nn.Module): latent encoder network
        p0_z (dist): prior distribution for latent code z
        device (device): torch device
    '''

    def __init__(self, decoder, encoder=None,
                 device=None):
        super().__init__()

        self.decoder = decoder.to(device)


        if encoder is not None:
            self.encoder = encoder.to(device)
        else:
            self.encoder = None

        self._device = device

    def forward(self, camera_ori_world, points_xy, ray_dir, img):
        ''' Performs a forward pass through the network.

        Args:
            p (tensor): sampled points
            inputs (tensor): conditioning input
            sample (bool): whether to sample for z
        '''
        c, c_local = self.encode_inputs(img)
        probs = self.decode(camera_ori_world, points_xy, ray_dir, c, c_local)  # (B, n_points, num_samples)
        return probs


    def encode_inputs(self, inputs):
        ''' Encodes the input.

        Args:
            input (tensor): the input
        '''

        if self.encoder is not None:
            c, c_local = self.encoder(inputs)
        else:
            # Return inputs?
            c, c_local = torch.empty(inputs.size(0), 0).to(torch.device('cuda:0'))

        return c, c_local

    def decode(self, camera_ori_world, points_xy, ray_dir, c_global, c_local):
        ''' Returns occupancy probabilities for the sampled points.

        Args:
            p (tensor): points
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''

        logits = self.decoder(camera_ori_world, points_xy, ray_dir, c_global, c_local)
        p_r = dist.Bernoulli(logits=logits)
        return p_r


    def to(self, device):
        ''' Puts the model to the device.

        Args:
            device (device): pytorch device
        '''
        model = super().to(device)
        model._device = device
        return model

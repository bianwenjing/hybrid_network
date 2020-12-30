import torch
import torch.nn as nn
from torch import distributions as dist
from im2mesh.onet_2d_2code.models import decoder

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

    def __init__(self, decoder, encoder_local=None, encoder_global=None,
                 device=None):
        super().__init__()

        self.decoder = decoder.to(device)


        if encoder_local is not None:
            self.encoder_local = encoder_local.to(device)
        else:
            self.encoder_local = None

        if encoder_global is not None:
            self.encoder = encoder_global.to(device)
        else:
            self.encoder = None

        self._device = device

    def forward(self, p, inputs, sample=True, **kwargs):
        ''' Performs a forward pass through the network.

        Args:
            p (tensor): sampled points
            inputs (tensor): conditioning input
            sample (bool): whether to sample for z
        '''
        batch_size = p.size(0)
        c, c_local = self.encode_inputs(inputs)
        p_r = self.decode(p, c, c_local, **kwargs)
        return p_r


    def encode_inputs(self, inputs):
        ''' Encodes the input.

        Args:
            input (tensor): the input
        '''

        if self.encoder is not None:
            c = self.encoder(inputs)
        else:
            # Return inputs?
            c = torch.empty(inputs.size(0), 0)
        if self.encoder_local is not None:
            c_local = self.encoder_local(inputs)
        else:
            # Return inputs?
            c_local = torch.empty(inputs.size(0), 0)



        return c, c_local

    def decode(self, p, c, c_local, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.

        Args:
            p (tensor): points
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''

        logits = self.decoder(p, c, c_local, **kwargs)
        # print('#########',logits.shape)
        p_r = dist.Bernoulli(logits=logits)
        # print('!!!!!!!!!!!1', type(p_r), p_r)
        return p_r


    def to(self, device):
        ''' Puts the model to the device.

        Args:
            device (device): pytorch device
        '''
        model = super().to(device)
        model._device = device
        return model

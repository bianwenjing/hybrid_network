import torch.nn as nn
from im2mesh.psgn_z.models.decoder import Decoder
from im2mesh.psgn.models.psgn_2branch import PCGN_2Branch

import torch
from torch import distributions as dist
from im2mesh.onet.models import encoder_latent, decoder

decoder_dict_psgn = {
    'simple': Decoder,
    'psgn_2branch': PCGN_2Branch
}

# Encoder latent dictionary
encoder_latent_dict = {
    'simple': encoder_latent.Encoder,
}

# Decoder dictionary
decoder_dict_oc = {
    'simple': decoder.Decoder,
    'cbatchnorm': decoder.DecoderCBatchNorm,
    'cbatchnorm2': decoder.DecoderCBatchNorm2,
    'batchnorm': decoder.DecoderBatchNorm,
    'cbatchnorm_noresnet': decoder.DecoderCBatchNormNoResnet,
}

class PCGN(nn.Module):
    r''' The Point Set Generation Network.

    For the PSGN, the input image is first passed to a encoder network,
    e.g. restnet-18 or the CNN proposed in the original publication. Next,
    this latent code is then used as the input for the decoder network, e.g.
    the 2-Branch model from the PSGN paper.

    Args:
        decoder (nn.Module): The decoder network
        encoder (nn.Module): The encoder network
    '''

    def __init__(self, decoder, encoder):
        super().__init__()
        self.decoder = decoder
        self.encoder = encoder

    def forward(self, x):
        c = self.encoder(x)
        points = self.decoder(c)
        return points

class CombinedNetwork(nn.Module):
    ''' Occupancy Network class.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        encoder_latent (nn.Module): latent encoder network
        p0_z (dist): prior distribution for latent code z
        device (device): torch device
    '''

    def __init__(self, decoder_oc, decoder_psgn, encoder=None, encoder_latent=None, p0_z=None,
                 device=None, consistency_loss=False):
        super().__init__()
        if p0_z is None:
            p0_z = dist.Normal(torch.tensor([]), torch.tensor([]))

        self.decoder_oc = decoder_oc.to(device)
        self.decoder_psgn = decoder_psgn.to(device)
        self.consistency_loss = consistency_loss

        if encoder_latent is not None:
            self.encoder_latent = encoder_latent.to(device)
        else:
            self.encoder_latent = None

        if encoder is not None:
            self.encoder = encoder.to(device)
        else:
            self.encoder = None

        self._device = device
        self.p0_z = p0_z

    def forward(self, p, inputs, sample=True, **kwargs):
        ''' Performs a forward pass through the network.

        Args:
            p (tensor): sampled points
            inputs (tensor): conditioning input
            sample (bool): whether to sample for z
        '''
        batch_size = p.size(0)
        c = self.encode_inputs(inputs)
        z = self.get_z_from_prior((batch_size,), sample=sample)
        p_r, p_r_psgn, points = self.decode(p, z, c, **kwargs)
        return p_r, p_r_psgn, points

    def compute_elbo(self, p, occ, inputs, **kwargs):
        ''' Computes the expectation lower bound.

        Args:
            p (tensor): sampled points
            occ (tensor): occupancy values for p
            inputs (tensor): conditioning input
        '''
        c = self.encode_inputs(inputs)
        q_z = self.infer_z(p, occ, c, **kwargs)
        z = q_z.rsample()
        p_r , p_r_psgn, points = self.decode(p, z, c, **kwargs)

        rec_error = -p_r.log_prob(occ).sum(dim=-1)
        kl = dist.kl_divergence(q_z, self.p0_z).sum(dim=-1)
        elbo = -rec_error - kl

        return elbo, rec_error, kl

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

        return c
    def decode_original(self, p, z, c, **kwargs):
        logits = self.decoder_oc(p, z, c, **kwargs)
        p_r = dist.Bernoulli(logits=logits)
        return p_r

    def decode(self, p, z, c, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.

        Args:
            p (tensor): points
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''
        points = self.decoder_psgn(z, c)
        logits = self.decoder_oc(p, z, c, **kwargs)
        p_r = dist.Bernoulli(logits=logits)
        if self.consistency_loss:
            logits_psgn = self.decoder_oc(points, z, c, **kwargs)
            p_r_psgn = dist.Bernoulli(logits=logits_psgn)
        else:
            p_r_psgn = None
        return p_r, p_r_psgn, points
    def decode2(self, z, c, **kwargs):
        '''
        decode use only psgn points
        '''
        points = self.decoder_psgn(z, c)
        logits_psgn = self.decoder_oc(points, z, c, **kwargs)
        p_r_psgn = dist.Bernoulli(logits=logits_psgn)
        return p_r_psgn, points


    def infer_z(self, p, occ, c, **kwargs):
        ''' Infers z.

        Args:
            p (tensor): points tensor
            occ (tensor): occupancy values for occ
            c (tensor): latent conditioned code c
        '''
        if self.encoder_latent is not None:
            mean_z, logstd_z = self.encoder_latent(p, occ, c, **kwargs)
        else:
            batch_size = p.size(0)
            mean_z = torch.empty(batch_size, 0).to(self._device)
            logstd_z = torch.empty(batch_size, 0).to(self._device)

        q_z = dist.Normal(mean_z, torch.exp(logstd_z))
        return q_z

    def get_z_from_prior(self, size=torch.Size([]), sample=True):
        ''' Returns z from prior distribution.

        Args:
            size (Size): size of z
            sample (bool): whether to sample
        '''
        if sample:
            z = self.p0_z.sample(size).to(self._device)
        else:
            z = self.p0_z.mean.to(self._device)
            z = z.expand(*size, *z.size())

        return z

    def to(self, device):
        ''' Puts the model to the device.

        Args:
            device (device): pytorch device
        '''
        model = super().to(device)
        model._device = device
        return model
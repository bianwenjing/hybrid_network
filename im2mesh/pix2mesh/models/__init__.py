import torch.nn as nn
from im2mesh.pix2mesh.models.decoder import Decoder
from im2mesh.pix2mesh.models.decoder2 import Decoder2


decoder_dict = {
    'simple': Decoder,
    'another': Decoder2
}


class Pix2Mesh(nn.Module):
    ''' Pixel2Mesh model.

    First, the input image is passed through a CNN to extract several feature
    maps. These feature maps as well as camera matrices are passed to the
    decoder to predict respective vertex locations of the output mesh

    '''
    def __init__(self, decoder, encoder):
        ''' Initialisation.

        Args:
            encoder (PyTorch model): The conditional network to obtain
                                     feature maps
            decoder (PyTorch model): The decoder network
        '''
        super().__init__()
        self.decoder = decoder
        self.nn_encoder = encoder

    def forward(self, x, camera_mat):
        fm = self.nn_encoder(x)
        pred = self.decoder(x, fm, camera_mat)
        return pred

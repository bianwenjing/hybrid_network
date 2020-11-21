import torch
import torch.nn as nn
import torch.nn.functional as F

from im2mesh.pix2mesh.layers2 import (GBottleneck, GConv)
from im2mesh.pix2mesh.gprojection import GProjection
# from im2mesh.pix2mesh.gpooling import GUnpooling


from im2mesh.pix2mesh.ellipsoid.mesh import Ellipsoid
from im2mesh.pix2mesh.layers import (
    GraphConvolution, GraphProjection, GraphUnpooling)

class Decoder2(nn.Module):

    def __init__(self, ellipsoid, device=None, hidden_dim=256,
                 feat_dim=1280, coor_dim=3, adjust_ellipsoid=False):
        super(Decoder2, self).__init__()

        self.hidden_dim = hidden_dim
        self.coord_dim = coor_dim
        self.last_hidden_dim = 128


        ellipsoid = Ellipsoid(0, ellipsoid)
        # if adjust_ellipsoid:
        #     ''' This is the inverse of the operation the Pixel2mesh authors'
        #     performed to original CAT model; it ensures that the ellipsoid
        #     has the same size and scale in the not-transformed coordinate
        #     system we are using. '''
        #     print("Adjusting ellipsoid.")
        #
        #     ellipsoid.coord = ellipsoid.coord / 0.57
        #     ellipsoid.coord[:, 1] = -ellipsoid.coord[:, 1]
        #     ellipsoid.coord[:, 2] = -ellipsoid.coord[:, 2]

        # self.init_pts = nn.Parameter(ellipsoid.coord, requires_grad=False)
        self.init_pts = ellipsoid.coord.cuda()


        self.gconv_activation = True
        self.nn_decoder = None

        # self.nn_encoder, self.nn_decoder = get_backbone(options)
        self.features_dim = feat_dim


        self.gcns = nn.ModuleList([
            GBottleneck(6, self.features_dim, self.hidden_dim, self.coord_dim,
                        ellipsoid.adj_mat[0], activation=self.gconv_activation),
            GBottleneck(6, self.features_dim + self.hidden_dim, self.hidden_dim, self.coord_dim,
                        ellipsoid.adj_mat[1], activation=self.gconv_activation),
            GBottleneck(6, self.features_dim + self.hidden_dim, self.hidden_dim, self.last_hidden_dim,
                        ellipsoid.adj_mat[2], activation=self.gconv_activation)
        ])

        # self.unpooling = nn.ModuleList([
        #     GUnpooling(ellipsoid.unpool_idx[0]),
        #     GUnpooling(ellipsoid.unpool_idx[1])
        # ])


        self.unpooling = nn.ModuleList([
            GraphUnpooling(ellipsoid.unpool_idx[0]),
            GraphUnpooling(ellipsoid.unpool_idx[1])
        ])


        # self.projection = GraphProjection()
        mesh_pos = [0, 0, 0]
        camera_f = [149.84375, 149.84375]
        # camera_f = [250, 250]
        camera_c = [68.5, 68.5]
        # camera_c = [112, 112]
        self.projection = GProjection(mesh_pos, camera_f, camera_c, bound=0,
                                      tensorflow_compatible=True)
        # self.projection = GraphProjection()


        self.gconv = GConv(in_features=self.last_hidden_dim, out_features=self.coord_dim,
                           adj_mat=ellipsoid.adj_mat[2])



    def forward(self, x, fm, camera_mat):
        batch_size = x.size(0)
        # img_feats = self.nn_encoder(img)
        img_shape = self.projection.image_feature_shape(x)

        init_pts = self.init_pts.data.unsqueeze(0).expand(batch_size, -1, -1)

        # GCN Block 1
        x = self.projection(img_shape, fm, init_pts)
        # x = self.projection(init_pts, fm, camera_mat)
        x1, x_hidden = self.gcns[0](x)


        # before deformation 2
        x1_up = self.unpooling[0](x1)

        # GCN Block 2
        x = self.projection(img_shape, fm, x1)
        # x = self.projection(x1, fm, camera_mat)
        x = self.unpooling[0](torch.cat([x, x_hidden], 2))
        # after deformation 2
        x2, x_hidden = self.gcns[1](x)

        # before deformation 3
        x2_up = self.unpooling[1](x2)

        # GCN Block 3
        x = self.projection(img_shape, fm, x2)
        # x = self.projection(x2, fm, camera_mat)
        x = self.unpooling[1](torch.cat([x, x_hidden], 2))
        x3, _ = self.gcns[2](x)
        if self.gconv_activation:
            x3 = F.relu(x3)
        # after deformation 3
        x3 = self.gconv(x3)

        # if self.nn_decoder is not None:
        #     reconst = self.nn_decoder(fm)
        # else:
        #     reconst = None

        # return {
        #     "pred_coord": [x1, x2, x3],
        #     "pred_coord_before_deform": [init_pts, x1_up, x2_up],
        #     "reconst": reconst
        # }
        outputs = (x1, x2, x3)
        outputs_2 = (init_pts, x1_up, x2_up)
        return outputs, outputs_2


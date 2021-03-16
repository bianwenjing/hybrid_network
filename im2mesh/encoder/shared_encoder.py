import torch.nn as nn
# import torch.nn.functional as F
import torchvision
import torch
from im2mesh.common import normalize_imagenet
import functools
import torch.nn.functional as F
class ResNet18Encoder(nn.Module):
    """
    2D (Spatial/Pixel-aligned/local) image encoder
    """

    def __init__(
        self,
        c_dim_local,
        c_dim_global,
        backbone="resnet18",
        pretrained=True,
        num_layers=4,
        index_interp="bilinear",
        index_padding="border",
        upsample_interp="bilinear",
        feature_scale=1.0,
        use_first_pool=True,
        norm_type="batch",
        normalize=True,
    ):
        """
        :param backbone Backbone network. Either custom, in which case
        model.custom_encoder.ConvEncoder is used OR resnet18/resnet34, in which case the relevant
        model from torchvision is used
        :param num_layers number of resnet layers to use, 1-5
        :param pretrained Whether to use model weights pretrained on ImageNet
        :param index_interp Interpolation to use for indexing
        :param index_padding Padding mode to use for indexing, border | zeros | reflection
        :param upsample_interp Interpolation to use for upscaling latent code
        :param feature_scale factor to scale all latent by. Useful (<1) if image
        is extremely large, to fit in memory.
        :param use_first_pool if false, skips first maxpool layer to avoid downscaling image
        features too much (ResNet only)
        :param norm_type norm type to applied; pretrained model must use batch
        """
        super().__init__()

        # self.use_custom_resnet = backbone == "custom"
        self.feature_scale = feature_scale
        self.use_first_pool = use_first_pool
        norm_layer = get_norm_layer(norm_type)
        self.c_dim_local = c_dim_local

        # self.model = getattr(torchvision.models, backbone)(
        #     pretrained=pretrained, norm_layer=norm_layer
        # )
        # pretrained = False
        self.features = getattr(torchvision.models, backbone)(
            pretrained=pretrained
        )
        # Following 2 lines need to be uncommented for older configs
        self.features.fc = nn.Sequential()
        # self.model.avgpool = nn.Sequential()
        self.latent_size = [0, 64, 128, 256, 512, 1024][num_layers]

        self.num_layers = num_layers
        self.index_interp = index_interp
        self.index_padding = index_padding
        self.upsample_interp = upsample_interp
        # self.register_buffer("latent", torch.empty(1, 1, 1, 1), persistent=False)
        # self.register_buffer(
        #     "latent_scaling", torch.empty(2, dtype=torch.float32), persistent=False
        # )
        # self.latent (B, L, H, W)

        self.fc = nn.Linear(512, 256)
        self.fc_global = nn.Linear(256, c_dim_global)
        self.normalize = normalize
        if c_dim_local!=0:
            self.conv_local = nn.Conv2d(512, c_dim_local, 1)

        # for name, p in self.named_parameters():
        #     if 'features' in name:
        #         p.requires_grad = False


    def forward(self, x):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, latent_size, H, W)
        """
        if self.normalize:
            x = normalize_imagenet(x)
        # print('))', torch.max(x), torch.min(x))

        x = self.features.conv1(x)
        # print(')))))))))', torch.max(x), torch.min(x))
        x = self.features.bn1(x)
        # print('***********', torch.max(x))
        x = self.features.relu(x)

        latents = [x]
        if self.num_layers > 1:
            if self.use_first_pool:
                x = self.features.maxpool(x)
            x = self.features.layer1(x)
            latents.append(x)
        x = self.features.layer2(x)
        if self.num_layers > 2:
            latents.append(x)
        x = self.features.layer3(x)
        if self.num_layers > 3:
            latents.append(x)
        x = self.features.layer4(x)
        if self.num_layers > 4:
            latents.append(x)
        x = self.features.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        global_feature = self.fc_global(x)

        # self.latents = latents
        if self.c_dim_local!=0:
            align_corners = None if self.index_interp == "nearest " else True
            latent_sz = latents[0].shape[-2:]
            for i in range(len(latents)):
                latents[i] = F.interpolate(
                    latents[i],
                    latent_sz,
                    mode=self.upsample_interp,
                    align_corners=align_corners,
                )

            latent = torch.cat(latents, dim=1)
            latent = self.conv_local(latent)
        else:
            latent = None
        return global_feature, latent

class ResNet34Encoder(nn.Module):
    """
    2D (Spatial/Pixel-aligned/local) image encoder
    """

    def __init__(
        self,
        c_dim_local,
        c_dim_global,
        backbone="resnet34",
        pretrained=True,
        num_layers=4,
        index_interp="bilinear",
        index_padding="border",
        upsample_interp="bilinear",
        feature_scale=1.0,
        use_first_pool=True,
        norm_type="batch",
        normalize=True,
    ):
        """
        :param backbone Backbone network. Either custom, in which case
        model.custom_encoder.ConvEncoder is used OR resnet18/resnet34, in which case the relevant
        model from torchvision is used
        :param num_layers number of resnet layers to use, 1-5
        :param pretrained Whether to use model weights pretrained on ImageNet
        :param index_interp Interpolation to use for indexing
        :param index_padding Padding mode to use for indexing, border | zeros | reflection
        :param upsample_interp Interpolation to use for upscaling latent code
        :param feature_scale factor to scale all latent by. Useful (<1) if image
        is extremely large, to fit in memory.
        :param use_first_pool if false, skips first maxpool layer to avoid downscaling image
        features too much (ResNet only)
        :param norm_type norm type to applied; pretrained model must use batch
        """
        super().__init__()

        if norm_type != "batch":
            assert not pretrained

        # self.use_custom_resnet = backbone == "custom"
        self.feature_scale = feature_scale
        self.use_first_pool = use_first_pool
        norm_layer = get_norm_layer(norm_type)
        self.c_dim_local = c_dim_local

        # self.model = getattr(torchvision.models, backbone)(
        #     pretrained=pretrained, norm_layer=norm_layer
        # )
        self.model = getattr(torchvision.models, backbone)(
            pretrained=pretrained
        )
        # Following 2 lines need to be uncommented for older configs
        # self.model.fc = nn.Sequential()
        # self.model.avgpool = nn.Sequential()
        self.latent_size = [0, 64, 128, 256, 512, 1024][num_layers]

        self.num_layers = num_layers
        self.index_interp = index_interp
        self.index_padding = index_padding
        self.upsample_interp = upsample_interp
        # self.register_buffer("latent", torch.empty(1, 1, 1, 1), persistent=False)
        # self.register_buffer(
        #     "latent_scaling", torch.empty(2, dtype=torch.float32), persistent=False
        # )
        # self.latent (B, L, H, W)

        self.fc = nn.Linear(512, c_dim_global)
        self.normalize = normalize
        if c_dim_local!=0:
            self.conv_local = nn.Conv2d(512, c_dim_local, 1)

        for name, p in self.named_parameters():
            if 'model' in name:
                p.requires_grad = False


    def forward(self, x):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, latent_size, H, W)
        """
        if self.normalize:
            x = normalize_imagenet(x)

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)

        latents = [x]
        if self.num_layers > 1:
            if self.use_first_pool:
                x = self.model.maxpool(x)
            x = self.model.layer1(x)
            latents.append(x)
        x = self.model.layer2(x)
        if self.num_layers > 2:
            latents.append(x)
        x = self.model.layer3(x)
        if self.num_layers > 3:
            latents.append(x)
        x = self.model.layer4(x)
        if self.num_layers > 4:
            latents.append(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        global_feature = self.fc(x)

        # self.latents = latents
        if self.c_dim_local!=0:
            align_corners = None if self.index_interp == "nearest " else True
            latent_sz = latents[0].shape[-2:]
            for i in range(len(latents)):
                latents[i] = F.interpolate(
                    latents[i],
                    latent_sz,
                    mode=self.upsample_interp,
                    align_corners=align_corners,
                )
            latent = torch.cat(latents, dim=1)
            latent = self.conv_local(latent)
        else:
            latent = None
        return global_feature, latent

    # @classmethod
    # def from_conf(cls, conf):
    #     return cls(
    #         conf.get_string("backbone"),
    #         pretrained=conf.get_bool("pretrained", True),
    #         num_layers=conf.get_int("num_layers", 4),
    #         index_interp=conf.get_string("index_interp", "bilinear"),
    #         index_padding=conf.get_string("index_padding", "border"),
    #         upsample_interp=conf.get_string("upsample_interp", "bilinear"),
    #         feature_scale=conf.get_float("feature_scale", 1.0),
    #         use_first_pool=conf.get_bool("use_first_pool", True),
    #     )

def get_norm_layer(norm_type="instance", group_norm_groups=32):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == "batch":
        norm_layer = functools.partial(
            nn.BatchNorm2d, affine=True, track_running_stats=True
        )
    elif norm_type == "instance":
        norm_layer = functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=False
        )
    elif norm_type == "group":
        norm_layer = functools.partial(nn.GroupNorm, group_norm_groups)
    elif norm_type == "none":
        norm_layer = None
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer
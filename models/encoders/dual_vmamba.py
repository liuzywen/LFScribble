import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from ..net_utils import FeatureFusionModule as FFM
from ..net_utils import FeatureRectifyModule as FRM
import math
import time
from engine.logger import get_logger
from models.encoders.vmamba import Backbone_VSSM, CrossMambaFusionBlock, ConcatMambaFusionBlock
import argparse

logger = get_logger()



class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x
class Decoder(nn.Module):
    def __init__(self, c):
        super(Decoder, self).__init__()
        self.pred = nn.Conv2d(c, 1, 1)
        self.conv1 = BasicConv2d(832, 64, 1)
        self.conv2 = BasicConv2d(832, 64, 1)
        self.conv3 = BasicConv2d(832, 64, 1)
        self.conv4 = BasicConv2d(832, 64, 1)
        self.conv_fuse_4_3 = BasicConv2d(128, 64, 1)
        self.conv_fuse_3_2 = BasicConv2d(128, 64, 1)
        self.conv_fuse_2_1 = BasicConv2d(128, 64, 1)

    def forward(self, f1, f2, f3, f4):
        f1 = self.conv1(f1)
        f2 = self.conv2(f2)
        f3 = self.conv3(f3)
        f4 = self.conv4(f4)
        f4 = F.interpolate(f4, scale_factor=2, mode='bilinear', align_corners=False)
        fuse4_3 = torch.cat((f4, f3), dim=1)
        fuse4_3 = self.conv_fuse_4_3(fuse4_3)
        fuse4_3 = F.interpolate(fuse4_3, scale_factor=2, mode='bilinear', align_corners=False)
        fuse3_2 = torch.cat((fuse4_3, f2), dim=1)
        fuse3_2 = self.conv_fuse_3_2(fuse3_2)
        fuse3_2 = F.interpolate(fuse3_2, scale_factor=2, mode='bilinear', align_corners=False)
        fuse2_1 = torch.cat((fuse3_2, f1), dim=1)
        fuse2_1 = self.conv_fuse_2_1(fuse2_1)
        fuse2_1 = F.interpolate(fuse2_1, scale_factor=4, mode='bilinear', align_corners=False)

        f = self.pred(fuse2_1)
        return f
class RGBXTransformer(nn.Module):
    def __init__(self, 
                 num_classes=1000,
                 norm_layer=nn.LayerNorm,
                 depths=[2,2,27,2], # [2,2,27,2] for vmamba small
                 dims=96,
                 pretrained=None,
                 mlp_ratio=4.0,
                 downsample_version='v1',
                 ape=False,
                 img_size=[480, 640],
                 patch_size=4,
                 drop_path_rate=0.2,
                 **kwargs):
        super().__init__()
        
        self.ape = ape

        self.vssm = Backbone_VSSM(
            pretrained=pretrained,
            norm_layer=norm_layer,
            num_classes=num_classes,
            depths=depths,
            dims=dims,
            mlp_ratio=mlp_ratio,
            downsample_version=downsample_version,
            drop_path_rate=drop_path_rate,
        )
        
        self.cross_mamba = nn.ModuleList(
            CrossMambaFusionBlock(
                hidden_dim=dims * (2 ** i),
                mlp_ratio=0.0,
                d_state=4,
            ) for i in range(4)
        )
        self.channel_attn_mamba = nn.ModuleList(
            ConcatMambaFusionBlock(
                hidden_dim=dims * (2 ** i),
                mlp_ratio=0.0,
                d_state=4,
            ) for i in range(4)
        )

        self.r_rfb4 = RFB_modified(1024, 64)
        self.r_rfb3 = RFB_modified(512, 64)
        self.r_rfb2 = RFB_modified(256, 64)
        self.r_rfb1 = RFB_modified(128, 64)

        self.f_rfb4 = RFB_modified(1024, 64)
        self.f_rfb3 = RFB_modified(512, 64)
        self.f_rfb2 = RFB_modified(256, 64)
        self.f_rfb1 = RFB_modified(128, 64)

        self.decoder = Decoder(64)

        # absolute position embedding
        if self.ape:
            self.patches_resolution = [img_size[0] // patch_size, img_size[1] // patch_size]
            self.absolute_pos_embed = []
            self.absolute_pos_embed_x = []
            for i_layer in range(len(depths)):
                input_resolution=(self.patches_resolution[0] // (2 ** i_layer),
                                      self.patches_resolution[1] // (2 ** i_layer))
                dim=int(dims * (2 ** i_layer))
                absolute_pos_embed = nn.Parameter(torch.zeros(1, dim, input_resolution[0], input_resolution[1]))
                trunc_normal_(absolute_pos_embed, std=.02)
                absolute_pos_embed_x = nn.Parameter(torch.zeros(1, dim, input_resolution[0], input_resolution[1]))
                trunc_normal_(absolute_pos_embed_x, std=.02)
                
                self.absolute_pos_embed.append(absolute_pos_embed)
                self.absolute_pos_embed_x.append(absolute_pos_embed_x)

    def forward_features(self, x_rgb, x_e):
        """
        x_rgb: B x C x H x W
        """
        ba = x_e.shape[0]
        
        outs_rgb = self.vssm(x_rgb) # B x C x H x W
        # torch.Size([B or B*12, 128, 64, 64])
        # torch.Size([B or B*12, 256, 32, 32])
        # torch.Size([B or B*12, 512, 16, 16])
        # torch.Size([B or B*12, 1024, 8, 8])
        outs_x = self.vssm(x_e) # B x C x H x W
        f_rgb = []
        f_focal = []
        for i in range(4):
            if self.ape:
                # this has been discarded
                out_rgb = self.absolute_pos_embed[i].to(outs_rgb[i].device) + outs_rgb[i]
                out_x = self.absolute_pos_embed_x[i].to(outs_x[i].device) + outs_x[i]
            else:
                out_rgb = outs_rgb[i]
                out_x = outs_x[i]
            f_rgb.append(out_rgb)
            f_focal.append(out_x)
        # rgb
        f_rgb[0] = self.r_rfb1(f_rgb[0])        # b 64 64 64
        f_rgb[1] = self.r_rfb2(f_rgb[1])        # b 64 32 32
        f_rgb[2] = self.r_rfb3(f_rgb[2])        # b 64 16 16
        f_rgb[3] = self.r_rfb4(f_rgb[3])        # b 64 8 8
        # focal
        x0f = self.f_rfb1(f_focal[0])
        x1f = self.f_rfb2(f_focal[1])
        x2f = self.f_rfb3(f_focal[2])
        x3f = self.f_rfb4(f_focal[3])
        x0f = torch.cat(torch.chunk(x0f.unsqueeze(1), ba, dim=0), dim=0)  # [12, ba, 32, 64, 64]
        ff1 = torch.cat(torch.chunk(x0f, 12, dim=0), dim=2).squeeze(0)  # [ba, 384, 64, 64]

        x1f = torch.cat(torch.chunk(x1f.unsqueeze(1), ba, dim=0), dim=0)  # [12, ba, 32, 32, 32]
        ff2 = torch.cat(torch.chunk(x1f, 12, dim=0), dim=2).squeeze(0)  # [ba, 384, 32, 32]

        x2f = torch.cat(torch.chunk(x2f.unsqueeze(1), ba, dim=0), dim=0)  # [12, ba, 32, 16, 16]
        ff3 = torch.cat(torch.chunk(x2f, 12, dim=0), dim=2).squeeze(0)  # [ba, 384, 16, 16]

        x3f = torch.cat(torch.chunk(x3f.unsqueeze(1), ba, dim=0), dim=0)  # [12, ba, 32, 8, 8]
        ff4 = torch.cat(torch.chunk(x3f, 12, dim=0), dim=2).squeeze(0)  # [ba, 384, 8, 8]

        fuse1 = torch.cat((ff1, f_rgb[0]), dim=1)       # b 832 64 64
        fuse2 = torch.cat((ff2, f_rgb[1]), dim=1)       # b 832 32 32
        fuse3 = torch.cat((ff3, f_rgb[2]), dim=1)       # b 832 16 16
        fuse4 = torch.cat((ff4, f_rgb[3]), dim=1)       # b 832 8 8

        pred = self.decoder(fuse1, fuse2, fuse3, fuse4)

        return pred

    def forward(self, x_rgb, x_e):
        out = self.forward_features(x_rgb, x_e)
        return out

class vssm_tiny(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(vssm_tiny, self).__init__(
            depths=[2, 2, 9, 2], 
            dims=96,
            pretrained='pretrained/vmamba/vssmtiny_dp01_ckpt_epoch_292.pth',
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.2,
        )

class vssm_small(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(vssm_small, self).__init__(
            depths=[2, 2, 27, 2],
            dims=96,
            pretrained='pretrained/vmamba/vssmsmall_dp03_ckpt_epoch_238.pth',
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.3,
        )

class vssm_base(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(vssm_base, self).__init__(
            depths=[2, 2, 27, 2],
            dims=128,
            img_size=[256, 256],
            ape=True,
            pretrained='pretrained/vmamba/vssmbase_dp06_ckpt_epoch_241.pth',
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.6, # VMamba-B with droppath 0.5 + no ema. VMamba-B* represents for VMamba-B with droppath 0.6 + ema
        )
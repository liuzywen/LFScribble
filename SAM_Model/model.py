import numpy as np
import torch
import os
import cv2
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from SAM_Model.adapter import Adapter
# from SAM_Model.build_sam import sam_model_registry
from SAM_Model.build_adsam import sam_model_registry
from SAM_Model.modeling_adsam.image_encoder import PostPosEmbed
from models.encoders.vmamba import Backbone_VSSM, MultimodalMamba, FusionMambaBlock, SSM, FocalMambaBlock
from timm.models.layers import DropPath, trunc_normal_


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

class Model(nn.Module):

    def __init__(
            self,
            cfg,
            embed_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = sam_model_registry[self.cfg.model_type](checkpoint=self.cfg.checkpoint)

        self.patch = nn.Conv2d(3, embed_dim, kernel_size=(16, 16), stride=(16, 16), padding=0)
        self.post_pos_embed = PostPosEmbed(embed_dim=embed_dim, ori_feature_size=1024 // 16,
                                           new_feature_size=16)  # new to sam

        # mamba
        self.multi_modal_mamba = MultimodalMamba(
            hidden_dim=128 * (2 ** 1),
            mlp_ratio=0.0,
            d_state=4,
        )

        self.fusion_mamba = FusionMambaBlock(
            hidden_dim=128 * (2 ** 1),
            mlp_ratio=0.0,
            d_state=4,
        )

        self.ssm = FocalMambaBlock(
            d_model=128 * (2 ** 1),
        )

        self.conv2d = nn.Conv2d(256,256,1)
        self.act = nn.SiLU()
        self.drop_path = DropPath(0)

        self.post_pos_embed2 = PostPosEmbed(embed_dim=embed_dim, ori_feature_size=1024 // 16,
                                           new_feature_size=16)  # new to sam
        self.all_conv = BasicConv2d(3328, 256, 1)
        self.all_conv2 = BasicConv2d(3072, 256, 1)
        self.depth = 24
        self.adapter_rgb_s = nn.ModuleList()
        self.adapter_depth_s = nn.ModuleList()
        for i in range(self.depth):
            adapter_r = Adapter(input_dim=embed_dim,
                                output_dim=128,  # 320
                                dropout=0.0,
                                adapter_scalar="learnable_scalar",
                                adapter_layernorm_option="in"
                                )
            adapter_d = Adapter(input_dim=embed_dim,
                                output_dim=128,  # 320
                                dropout=0.0,
                                adapter_scalar="learnable_scalar",
                                adapter_layernorm_option="in"
                                )
            self.adapter_rgb_s.append(adapter_r)
            self.adapter_depth_s.append(adapter_d)


    def setup(self):
        if self.cfg.freeze_image_encoder:
            print("冻结编码器")
            for param in self.model.image_encoder.parameters():
                param.requires_grad_(False)

        if self.cfg.freeze_prompt_encoder:
            print("冻结提示编码器")
            for name, param in self.model.prompt_encoder.named_parameters():
                # print("ppp", name)
                param.requires_grad_(False)
        if self.cfg.freeze_mask_decoder:
            print("冻结解码器")
            for name, param in self.model.mask_decoder.named_parameters():
                param.requires_grad_(False)

    def forward(self, focal, all_f):

        batch, C, H, W = focal.shape
        ba = batch // 12

        rgb_embeddings = self.model.image_encoder.patch_embed(focal)
        all_f_embeddings = self.model.image_encoder.patch_embed(all_f)

        if self.model.image_encoder.pos_embed is not None:
            rgb_embeddings = rgb_embeddings + self.post_pos_embed(self.model.image_encoder.pos_embed)
            all_f_embeddings = all_f_embeddings + self.post_pos_embed2(self.model.image_encoder.pos_embed)

        t = 0
        for i, blk in enumerate(self.model.image_encoder.blocks):
            rgb_embeddings = blk(rgb_embeddings) + self.adapter_rgb_s[t](rgb_embeddings)
            all_f_embeddings = blk(all_f_embeddings) + self.adapter_depth_s[t](all_f_embeddings)
            t = t + 1

        rgb_embeddings = self.model.image_encoder.neck(rgb_embeddings.permute(0, 3, 1, 2))
        all_f_embeddings = self.model.image_encoder.neck(all_f_embeddings.permute(0, 3, 1, 2))
        ssm_rgb_embeddings = self.ssm(rgb_embeddings.permute(0, 2, 3, 1).contiguous())
        ssm_rgb_embeddings = ssm_rgb_embeddings.permute(0, 3, 1, 2)
        if ba == 2:
            first_half_mean = torch.mean(ssm_rgb_embeddings[:12], dim=0, keepdim=True)
            second_half_mean = torch.mean(ssm_rgb_embeddings[12:], dim=0, keepdim=True)
            ssm_rgb_embeddings = torch.cat([first_half_mean, second_half_mean], dim=0)
        else:
            ssm_rgb_embeddings = torch.mean(ssm_rgb_embeddings[:12], dim=0, keepdim=True)
        out_rgb = all_f_embeddings
        out_f = ssm_rgb_embeddings
        cross_rgb, cross_x = self.multi_modal_mamba(out_rgb.permute(0, 2, 3, 1).contiguous(),
                                              out_f.permute(0, 2, 3, 1).contiguous())  # B x H x W x C
        x_fuse = self.fusion_mamba(cross_rgb, cross_x).permute(0, 3, 1, 2).contiguous()
        fuse_embeddings = x_fuse
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
        )

        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=fuse_embeddings,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        masks = F.interpolate(
            low_res_masks,
            (H, W),
            mode="bilinear",
            align_corners=False,
        )

        return masks

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .base_model import BaseModel
from .blocks import (
    FeatureFusionBlock,
    FeatureFusionBlock_custom,
    Interpolate,
    _make_encoder,
    forward_vit,
)


class VideoAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., seq_len=12, relative_position_encoding=True):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.seq_len = seq_len
        self.relative_position_encoding = relative_position_encoding
        self.relative_position_bias_table = nn.Parameter(torch.zeros(2 * seq_len - 1, num_heads))  # 2*T-1, nH
        self.register_buffer("relative_position_index", self.get_position_index())  # T, T

    def get_position_index(self):
        coords = torch.arange(self.seq_len)  # T
        relative_coords = coords[:, None] - coords[None, :]  # T, T
        relative_coords += self.seq_len - 1  # shift to start from 0
        relative_position_index = relative_coords  # T, T
        return relative_position_index

    def forward(self, x):
        BT, N, C = x.shape
        T = self.seq_len
        B = BT // T
        assert BT == B * T

        x = rearrange(x, '(b t) n c -> b (t n) c', b=B, t=T)

        qkv = self.qkv(x).reshape(B, T * N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, nH, THW, THW

        if self.relative_position_encoding:
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.reshape(-1)].reshape(T, T, -1)  # T, T, nH

            relative_position_bias = relative_position_bias[None].permute(3, 0, 1, 2).expand(self.num_heads, N*N, T, T)
            relative_position_bias = F.pixel_shuffle(relative_position_bias, upscale_factor=N).permute(1, 0, 2, 3)

            attn = attn + relative_position_bias  # B, nH, THW, THW

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, T * N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = rearrange(x, 'b (t n) c -> (b t) n c', t=T, n=N)

        return x

def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )


class DPT(BaseModel):
    def __init__(
        self,
        head,
        features=256,
        backbone="vitb_rn50_384",
        readout="project",
        channels_last=False,
        use_bn=False,
        enable_attention_hooks=False,
    ):

        super(DPT, self).__init__()

        self.channels_last = channels_last

        hooks = {
            "vitb_rn50_384": [0, 1, 8, 11],
            "vitb16_384": [2, 5, 8, 11],
            "vitl16_384": [5, 11, 17, 23],
        }

        # Instantiate backbone and reassemble blocks
        self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            False,  # Set to true of you want to train from scratch, uses ImageNet weights
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks[backbone],
            use_readout=readout,
            enable_attention_hooks=enable_attention_hooks,
        )

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.scratch.output_conv = head

    def forward(self, x):
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv(path_1)

        return out


class DPTDepthModel(DPT):
    def __init__(
        self, path=None, non_negative=True, scale=1.0, shift=0.0, invert=False,
            seq_len=4, relative_position_encoding=True, attn_interval=2, **kwargs
    ):
        features = kwargs["features"] if "features" in kwargs else 256

        self.scale = scale
        self.shift = shift
        self.invert = invert

        head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Identity(),
        )

        super().__init__(head, **kwargs)

        if path is not None:
            self.load(path)

        for i in range(0, len(self.pretrained.model.blocks), attn_interval):
            attn_ori = self.pretrained.model.blocks[i].attn
            attn_new = VideoAttention(
                dim=attn_ori.qkv.in_features,
                num_heads=attn_ori.num_heads,
                qkv_bias=attn_ori.qkv.bias is not None,
                attn_drop=attn_ori.attn_drop.p,
                proj_drop=attn_ori.proj_drop.p,
                seq_len=seq_len,
                relative_position_encoding=relative_position_encoding,
            )

            attn_new.qkv.weight.data.copy_(attn_ori.qkv.weight.clone())
            attn_new.proj.weight.data.copy_(attn_ori.proj.weight.clone())
            self.pretrained.model.blocks[i].attn = attn_new

    def forward(self, x):
        B, T, C, H, W = x.shape

        inv_depth = super().forward(x.flatten(0, 1)).squeeze(dim=1)

        if self.invert:
            depth = self.scale * inv_depth + self.shift
            depth[depth < 1e-8] = 1e-8
            depth = 1.0 / depth
            return depth.reshape(B, T, 1, H, W)
        else:
            return inv_depth.reshape(B, T, 1, H, W)

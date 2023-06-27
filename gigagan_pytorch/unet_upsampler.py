import math
from math import log2
from random import random
from functools import partial

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from gigagan_pytorch.attend import Attend


# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def identity(t, *args, **kwargs):
    return t

def is_power_of_two(n):
    return log2(n).is_integer()

# small helper modules

class Upsample(nn.Module):
    def __init__(self, dim, dim_out = None):
        super().__init__()
        dim_out = default(dim_out, dim)

        conv = nn.Conv2d(dim, dim_out * 4, 1)
        self.init_conv_(conv)

        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            nn.PixelShuffle(2)
        )

    def init_conv_(self, conv):
        o, *rest_shape = conv.weight.shape
        conv_weight = torch.empty(o // 4, *rest_shape)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o 4) ...')

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        return self.net(x)

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        flash = False
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.attend = Attend(flash = flash)

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), qkv)

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# model

class UnetUpsampler(nn.Module):
    def __init__(
        self,
        dim,
        *,
        image_size,
        input_image_size,
        init_dim = None,
        out_dim = None,
        dim_mults = (1, 2, 4, 8),
        channels = 3,
        resnet_block_groups = 8,
        full_attn = (False, False, False, True),
        flash_attn = True,
        resize_mode = 'bilinear'
    ):
        super().__init__()

        assert is_power_of_two(image_size) and is_power_of_two(input_image_size), 'both output image size and input image size must be power of 2'
        assert input_image_size < image_size, 'input image size must be smaller than the output image size, thus upsampling'

        num_layer_no_downsample = int(log2(image_size) - log2(input_image_size))
        assert num_layer_no_downsample <= len(dim_mults), 'you need more stages in this unet for the level of upsampling'

        self.image_size = image_size
        self.input_image_size = input_image_size

        # determine dimensions

        self.channels = channels
        input_channels = channels

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]

        *_, mid_dim = dims

        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # attention

        full_attn = cast_tuple(full_attn, length = len(dim_mults))
        assert len(full_attn) == len(dim_mults)

        FullAttention = partial(Attention, flash = flash_attn)

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), layer_full_attn) in enumerate(zip(in_out, full_attn)):
            is_last = ind >= (num_resolutions - 1)
            should_not_downsample = ind < num_layer_no_downsample

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in),
                block_klass(dim_in, dim_in),
                attn_klass(dim_in),
                Downsample(dim_in, dim_out) if not should_not_downsample else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        self.mid_block1 = block_klass(mid_dim, mid_dim)
        self.mid_attn = FullAttention(mid_dim)
        self.mid_block2 = block_klass(mid_dim, mid_dim)
        self.mid_to_rgb = nn.Conv2d(mid_dim, channels, 1)

        for ind, ((dim_in, dim_out), layer_full_attn) in enumerate(zip(reversed(in_out), reversed(full_attn))):
            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.ups.append(nn.ModuleList([
                Upsample(dim_out, dim_in),
                Upsample(channels),
                nn.Conv2d(dim_in, channels, 1),
                block_klass(dim_in * 2, dim_in),
                block_klass(dim_in * 2, dim_in),
                attn_klass(dim_in),
            ]))

        self.out_dim = default(out_dim, channels)

        self.final_res_block = block_klass(dim, dim)
        self.final_to_rgb = nn.Conv2d(dim, channels, 1)

        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

        # resize mode

        self.resize_mode = resize_mode

    @property
    def device(self):
        return next(self.parameters()).device

    def resize_image_to(self, x, size):
        return F.interpolate(x, (size, size), mode = self.resize_mode)

    def forward(
        self,
        x,
        return_all_rgbs = False
    ):
        shape = x.shape
        assert shape[-2:] == ((self.input_image_size,) * 2)

        x = self.init_conv(x)

        h = []

        # downsample stages

        for block1, block2, attn, downsample in self.downs:
            x = block1(x)
            h.append(x)

            x = block2(x)
            x = attn(x) + x
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x)

        # rgbs

        rgbs = []

        init_rgb_shape = list(x.shape)
        init_rgb_shape[1] = self.channels

        rgb = self.mid_to_rgb(x)
        rgbs.append(rgb)

        # upsample stages

        for upsample, upsample_rgb, to_rgb, block1, block2, attn in self.ups:

            x = upsample(x)
            rgb = upsample_rgb(rgb)

            res1 = h.pop()
            res2 = h.pop()

            fmap_size = x.shape[-1]
            residual_fmap_size = res1.shape[-1]

            if residual_fmap_size != fmap_size:
                res1 = self.resize_image_to(res1, fmap_size)
                res2 = self.resize_image_to(res2, fmap_size)

            x = torch.cat((x, res1), dim = 1)
            x = block1(x)

            x = torch.cat((x, res2), dim = 1)
            x = block2(x)

            x = attn(x) + x

            rgb = rgb + to_rgb(x)
            rgbs.append(rgb)

        x = self.final_res_block(x)

        rgb = rgb + self.final_to_rgb(x)

        if not return_all_rgbs:
            return rgb

        # only keep those rgbs whose feature map is greater than the input image to be upsampled

        rgbs = list(filter(lambda t: t.shape[-1] >= shape[-1], rgbs))

        return rgb, rgbs

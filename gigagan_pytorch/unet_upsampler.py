from math import log2
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from gigagan_pytorch.attend import Attend
from gigagan_pytorch.gigagan_pytorch import (
    BaseGenerator,
    StyleNetwork,
    AdaptiveConv2DMod,
    TextEncoder,
    CrossAttentionBlock,
    Upsample
)

from beartype import beartype
from beartype.typing import Optional, List, Union, Dict, Iterable

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

def null_iterator():
    while True:
        yield None

# small helper modules

class PixelShuffleUpsample(nn.Module):
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
    def __init__(
        self,
        dim,
        dim_out,
        groups = 8,
        num_conv_kernels = 0
    ):
        super().__init__()
        self.proj = AdaptiveConv2DMod(dim, dim_out, kernel = 3, num_conv_kernels = num_conv_kernels)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(
        self,
        x,
        conv_mods_iter: Optional[Iterable] = None
    ):
        conv_mods_iter = default(conv_mods_iter, null_iterator())

        x = self.proj(
            x,
            mod = next(conv_mods_iter),
            kernel_mod = next(conv_mods_iter)
        )

        x = self.norm(x)
        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        *,
        groups = 8,
        num_conv_kernels = 0,
        style_dims: List = []
    ):
        super().__init__()
        style_dims.extend([
            dim,
            num_conv_kernels,
            dim_out,
            num_conv_kernels
        ])

        self.block1 = Block(dim, dim_out, groups = groups, num_conv_kernels = num_conv_kernels)
        self.block2 = Block(dim_out, dim_out, groups = groups, num_conv_kernels = num_conv_kernels)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(
        self,
        x,
        conv_mods_iter: Optional[Iterable] = None
    ):
        h = self.block1(x, conv_mods_iter = conv_mods_iter)
        h = self.block2(h, conv_mods_iter = conv_mods_iter)

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

# feedforward

def FeedForward(dim, mult = 4):
    return nn.Sequential(
        RMSNorm(dim),
        nn.Conv2d(dim, dim * mult, 1),
        nn.GELU(),
        nn.Conv2d(dim * mult, dim, 1)
    )

# transformers

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        depth = 1,
        flash_attn = True,
        ff_mult = 4
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads, flash = flash_attn),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return x

class LinearTransformer(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        depth = 1,
        ff_mult = 4
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                LinearAttention(dim = dim, dim_head = dim_head, heads = heads),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return x

# model

class UnetUpsampler(BaseGenerator):

    @beartype
    def __init__(
        self,
        dim,
        *,
        image_size,
        input_image_size,
        init_dim = None,
        out_dim = None,
        text_encoder: Optional[Union[TextEncoder, Dict]] = None,
        style_network: Optional[Union[StyleNetwork, Dict]] = None,
        style_network_dim = None,
        dim_mults = (1, 2, 4, 8),
        channels = 3,
        resnet_block_groups = 8,
        full_attn = (False, False, False, True),
        cross_attn = (False, False, False, True),
        flash_attn = True,
        self_attn_dim_head = 64,
        self_attn_heads = 8,
        self_attn_dot_product = True,
        self_attn_ff_mult = 4,
        attn_depths = (1, 1, 1, 1),
        cross_attn_dim_head = 64,
        cross_attn_heads = 8,
        cross_ff_mult = 4,
        mid_attn_depth = 1,
        num_conv_kernels = 2,
        resize_mode = 'bilinear',
        unconditional = True
    ):
        super().__init__()

        # style network

        if isinstance(text_encoder, dict):
            text_encoder = TextEncoder(**text_encoder)

        self.text_encoder = text_encoder

        if isinstance(style_network, dict):
            style_network = StyleNetwork(**style_network)

        self.style_network = style_network

        assert exists(style_network) ^ exists(style_network_dim), 'either style_network or style_network_dim must be passed in'

        # validate text conditioning and style network hparams

        self.unconditional = unconditional
        assert unconditional ^ exists(text_encoder), 'if unconditional, text encoder should not be given, and vice versa'
        assert not (unconditional and exists(style_network) and style_network.dim_text_latent > 0)
        assert unconditional or text_encoder.dim == style_network.dim_text_latent, 'the `dim_text_latent` on your StyleNetwork must be equal to the `dim` set for the TextEncoder'

        assert is_power_of_two(image_size) and is_power_of_two(input_image_size), 'both output image size and input image size must be power of 2'
        assert input_image_size < image_size, 'input image size must be smaller than the output image size, thus upsampling'

        num_layer_no_downsample = int(log2(image_size) - log2(input_image_size))
        assert num_layer_no_downsample <= len(dim_mults), 'you need more stages in this unet for the level of upsampling'

        self.image_size = image_size
        self.input_image_size = input_image_size

        # setup adaptive conv

        style_embed_split_dims = []

        # determine dimensions

        self.channels = channels
        input_channels = channels

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]

        *_, mid_dim = dims

        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(
            ResnetBlock,
            groups = resnet_block_groups,
            num_conv_kernels = num_conv_kernels,
            style_dims = style_embed_split_dims
        )

        # attention

        full_attn = cast_tuple(full_attn, length = len(dim_mults))
        assert len(full_attn) == len(dim_mults)

        FullAttention = partial(Transformer, flash_attn = flash_attn)

        cross_attn = cast_tuple(cross_attn, length = len(dim_mults))
        assert unconditional or len(full_attn) == len(dim_mults)

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_cross_attn, layer_attn_depth) in enumerate(zip(in_out, full_attn, cross_attn, attn_depths)):
            ind >= (num_resolutions - 1)

            should_not_downsample = ind < num_layer_no_downsample
            has_cross_attn = not self.unconditional and layer_cross_attn

            attn_klass = FullAttention if layer_full_attn else LinearTransformer

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in),
                block_klass(dim_in, dim_in),
                CrossAttentionBlock(dim_in, dim_context = text_encoder.dim, dim_head = self_attn_dim_head, heads = self_attn_heads, ff_mult = self_attn_ff_mult) if has_cross_attn else None,
                attn_klass(dim_in, dim_head = self_attn_dim_head, heads = self_attn_heads, depth = layer_attn_depth),
                Downsample(dim_in, dim_out) if not should_not_downsample else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        self.mid_block1 = block_klass(mid_dim, mid_dim)
        self.mid_attn = FullAttention(mid_dim, dim_head = self_attn_dim_head, heads = self_attn_heads, depth = mid_attn_depth)
        self.mid_block2 = block_klass(mid_dim, mid_dim)
        self.mid_to_rgb = nn.Conv2d(mid_dim, channels, 1)

        for ind, ((dim_in, dim_out), layer_cross_attn, layer_full_attn, layer_attn_depth) in enumerate(zip(reversed(in_out), reversed(full_attn), reversed(cross_attn), reversed(attn_depths))):

            attn_klass = FullAttention if layer_full_attn else LinearTransformer
            has_cross_attn = not self.unconditional and layer_cross_attn

            self.ups.append(nn.ModuleList([
                PixelShuffleUpsample(dim_out, dim_in),
                Upsample(),
                nn.Conv2d(dim_in, channels, 1),
                block_klass(dim_in * 2, dim_in),
                block_klass(dim_in * 2, dim_in),
                CrossAttentionBlock(dim_in, dim_context = text_encoder.dim, dim_head = self_attn_dim_head, heads = self_attn_heads, ff_mult = cross_ff_mult) if has_cross_attn else None,
                attn_klass(dim_in, dim_head = cross_attn_dim_head, heads = self_attn_heads, depth = layer_attn_depth),
            ]))

        self.out_dim = default(out_dim, channels)

        self.final_res_block = block_klass(dim, dim)

        self.final_to_rgb = nn.Conv2d(dim, channels, 1)

        # resize mode

        self.resize_mode = resize_mode

        # determine the projection of the style embedding to convolutional modulation weights (+ adaptive kernel selection weights) for all layers

        self.style_to_conv_modulations = nn.Linear(style_network.dim, sum(style_embed_split_dims))
        self.style_embed_split_dims = style_embed_split_dims

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def total_params(self):
        return sum([p.numel() for p in self.parameters()])

    def resize_image_to(self, x, size):
        return F.interpolate(x, (size, size), mode = self.resize_mode)

    def forward(
        self,
        lowres_image,
        styles = None,
        noise = None,
        texts: Optional[List[str]] = None,
        global_text_tokens = None,
        fine_text_tokens = None,
        text_mask = None,
        return_all_rgbs = False,
        replace_rgb_with_input_lowres_image = True   # discriminator should also receive the low resolution image the upsampler sees
    ):
        x = lowres_image
        shape = x.shape
        batch_size = shape[0]

        assert shape[-2:] == ((self.input_image_size,) * 2)

        # take care of text encodings
        # which requires global text tokens to adaptively select the kernels from the main contribution in the paper
        # and fine text tokens to attend to using cross attention

        if not self.unconditional:
            if exists(texts):
                assert exists(self.text_encoder)
                global_text_tokens, fine_text_tokens, text_mask = self.text_encoder(texts)
            else:
                assert all([*map(exists, (global_text_tokens, fine_text_tokens, text_mask))])
        else:
            assert not any([*map(exists, (texts, global_text_tokens, fine_text_tokens))])

        # styles

        if not exists(styles):
            assert exists(self.style_network)

            noise = default(noise, torch.randn((batch_size, self.style_network.dim), device = self.device))
            styles = self.style_network(noise, global_text_tokens)

        # project styles to conv modulations

        conv_mods = self.style_to_conv_modulations(styles)
        conv_mods = conv_mods.split(self.style_embed_split_dims, dim = -1)
        conv_mods = iter(conv_mods)

        # initial conv

        x = self.init_conv(x)

        h = []

        # downsample stages

        for block1, block2, cross_attn, attn, downsample in self.downs:
            x = block1(x, conv_mods_iter = conv_mods)
            h.append(x)

            x = block2(x, conv_mods_iter = conv_mods)

            x = attn(x)

            if exists(cross_attn):
                x = cross_attn(x, context = fine_text_tokens, mask = text_mask)

            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, conv_mods_iter = conv_mods)
        x = self.mid_attn(x)
        x = self.mid_block2(x, conv_mods_iter = conv_mods)

        # rgbs

        rgbs = []

        init_rgb_shape = list(x.shape)
        init_rgb_shape[1] = self.channels

        rgb = self.mid_to_rgb(x)
        rgbs.append(rgb)

        # upsample stages

        for upsample, upsample_rgb, to_rgb, block1, block2, cross_attn, attn in self.ups:

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
            x = block1(x, conv_mods_iter = conv_mods)

            x = torch.cat((x, res2), dim = 1)
            x = block2(x, conv_mods_iter = conv_mods)

            if exists(cross_attn):
                x = cross_attn(x, context = fine_text_tokens, mask = text_mask)

            x = attn(x)

            rgb = rgb + to_rgb(x)
            rgbs.append(rgb)

        x = self.final_res_block(x, conv_mods_iter = conv_mods)

        assert len([*conv_mods]) == 0

        rgb = rgb + self.final_to_rgb(x)

        if not return_all_rgbs:
            return rgb

        # only keep those rgbs whose feature map is greater than the input image to be upsampled

        rgbs = list(filter(lambda t: t.shape[-1] >= shape[-1], rgbs))

        if not replace_rgb_with_input_lowres_image:
            return rgb, rgbs

        # replace the rgb of the corresponding same dimension as the input low res image

        output_rgbs = []

        for rgb in rgbs:
            if rgb.shape[-1] == lowres_image.shape[-1]:
                output_rgbs.append(lowres_image)
            else:
                output_rgbs.append(rgb)

        return rgb, output_rgbs

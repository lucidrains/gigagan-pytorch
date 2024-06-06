from __future__ import annotations

from math import log2
from functools import partial
from itertools import islice

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

from gigagan_pytorch.attend import Attend
from gigagan_pytorch.gigagan_pytorch import (
    BaseGenerator,
    StyleNetwork,
    AdaptiveConv2DMod,
    AdaptiveConv1DMod,
    TextEncoder,
    CrossAttentionBlock,
    Upsample,
    PixelShuffleUpsample,
    Blur
)

from kornia.filters import filter3d, filter2d

from beartype import beartype
from beartype.typing import List, Dict, Iterable, Literal

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

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

def fold_space_into_batch(x):
    x = rearrange(x, 'b c t h w -> b h w c t')
    x, ps = pack_one(x, '* c t')

    def split_space_from_batch(out):
        out = unpack_one(x, ps, '* c t')
        out = rearrange(out, 'b h w c t -> b c t h w')
        return out

    return x, split_space_from_batch

# small helper modules

def interpolate_1d(x, length, mode = 'bilinear'):
    x = rearrange(x, 'b c t -> b c t 1')
    x = F.interpolate(x, (length, 1), mode = mode)
    return rearrange(x, 'b c t 1 -> b c t')

class Downsample(Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        skip_downsample = False,
        has_temporal_layers = False
    ):
        super().__init__()
        dim_out = default(dim_out, dim)

        self.skip_downsample = skip_downsample

        self.conv2d = nn.Conv2d(dim, dim_out, 3, padding = 1)

        self.has_temporal_layers = has_temporal_layers

        if has_temporal_layers:
            self.conv1d = nn.Conv1d(dim_out, dim_out, 3, padding = 1)

            nn.init.dirac_(self.conv1d.weight)
            nn.init.zeros_(self.conv1d.bias)

        self.register_buffer('filter', torch.Tensor([1., 2., 1.]))

    def forward(self, x):
        batch = x.shape[0]
        is_input_video = x.ndim == 5

        assert not (is_input_video and not self.has_temporal_layers)

        if is_input_video:
            x = rearrange(x, 'b c t h w -> (b t) c h w')

        x = self.conv2d(x)

        if is_input_video:
            x = rearrange(x, '(b t) c h w -> b h w c t', b = batch)
            x, ps = pack_one(x, '* c t')

            x = self.conv1d(x)

            x = unpack_one(x, ps, '* c t')
            x = rearrange(x, 'b h w c t -> b c t h w')

        # if not downsampling, early return

        if self.skip_downsample:
            return x, x[:, 0:0]

        # save before blur to subtract out for high frequency fmap skip connection

        before_blur_input = x

        # blur 2d or 3d, depending

        f = self.filter

        if is_input_video:
            f = f[None, None, None, :] * f[None, None, :, None] * f[None, :, None, None]
            filter_fn = filter3d
            maxpool_fn = F.max_pool3d
        else:
            f = f[None, None, :] * f[None, :, None]
            filter_fn = filter2d
            maxpool_fn = F.max_pool2d

        blurred = filter_fn(x, f, normalized = True)

        # get high frequency fmap

        high_freq_fmap = before_blur_input - blurred

        # max pool 2d or 3d, depending

        x = maxpool_fn(x, kernel_size = 2)

        return x, high_freq_fmap

class TemporalBlur(Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)

    def forward(self, x):
        f = repeat(self.f, 't -> 1 t h w', h = 3, w = 3)
        return filter3d(x, f, normalized = True)

class TemporalUpsample(Module):
    def __init__(
        self,
        dim,
        dim_out = None
    ):
        super().__init__()
        self.blur = TemporalBlur()

    def forward(self, x):
        assert x.ndim == 5
        time = x.shape[2]

        x = rearrange(x, 'b c t h w -> b h w c t')
        x, ps = pack_one(x, '* c t')

        x = interpolate_1d(x, time * 2, mode = 'bilinear')

        x = unpack_one(x, ps, '* c t')
        x = rearrange(x, 'b h w c t -> b c t h w')
        x = self.blur(x)
        return x

class PixelShuffleTemporalUpsample(Module):
    def __init__(self, dim, dim_out = None):
        super().__init__()
        dim_out = default(dim_out, dim)

        conv = nn.Conv3d(dim, dim_out * 2, 1)

        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            Rearrange('b (c p) t h w -> b c (t p) h w', p = 2)
        )

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, t, h, w = conv.weight.shape
        conv_weight = torch.empty(o // 2, i, t, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o 2) ...')

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        return self.net(x)

# norm

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        spatial_dims = ((1,) * (x.ndim - 2))
        gamma = self.gamma.reshape(-1, *spatial_dims)

        return F.normalize(x, dim = 1) * gamma * self.scale

# building block modules

class Block(Module):
    @beartype
    def __init__(
        self,
        dim,
        dim_out,
        num_conv_kernels = 0,
        conv_type: Literal['1d', '2d'] = '2d',
    ):
        super().__init__()

        adaptive_conv_klass = AdaptiveConv2DMod if conv_type == '2d' else AdaptiveConv1DMod

        self.proj = adaptive_conv_klass(dim, dim_out, kernel = 3, num_conv_kernels = num_conv_kernels)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()

    def forward(
        self,
        x,
        conv_mods_iter: Iterable | None = None
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

class ResnetBlock(Module):
    @beartype
    def __init__(
        self,
        dim,
        dim_out,
        *,
        num_conv_kernels = 0,
        conv_type: Literal['1d', '2d'] = '2d',
        style_dims: List[int] = []
    ):
        super().__init__()

        mod_dims = [
            dim,
            num_conv_kernels,
            dim_out,
            num_conv_kernels
        ]

        style_dims.extend(mod_dims)

        self.num_mods = len(mod_dims)

        self.block1 = Block(dim, dim_out, num_conv_kernels = num_conv_kernels, conv_type = conv_type)
        self.block2 = Block(dim_out, dim_out, num_conv_kernels = num_conv_kernels, conv_type = conv_type)

        conv_klass = nn.Conv2d if conv_type == '2d' else nn.Conv1d
        self.res_conv = conv_klass(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(
        self,
        x,
        conv_mods_iter: Iterable | None = None
    ):
        h = self.block1(x, conv_mods_iter = conv_mods_iter)
        h = self.block2(h, conv_mods_iter = conv_mods_iter)

        return h + self.res_conv(x)

class LinearAttention(Module):
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

class Attention(Module):
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

class Transformer(Module):
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
        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads, flash = flash_attn),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return x

class LinearTransformer(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        depth = 1,
        ff_mult = 4
    ):
        super().__init__()
        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
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
        text_encoder: TextEncoder | Dict | None = None,
        style_network: StyleNetwork | Dict | None = None,
        style_network_dim = None,
        dim_mults = (1, 2, 4, 8, 16),
        channels = 3,
        full_attn = (False, False, False, True, True),
        cross_attn = (False, False, False, True, True),
        flash_attn = True,
        self_attn_dim_head = 64,
        self_attn_heads = 8,
        self_attn_dot_product = True,
        self_attn_ff_mult = 4,
        attn_depths = (1, 1, 1, 1, 1),
        temporal_attn_depths = (1, 1, 1, 1, 1),
        cross_attn_dim_head = 64,
        cross_attn_heads = 8,
        cross_ff_mult = 4,
        has_temporal_layers = False,
        mid_attn_depth = 1,
        num_conv_kernels = 2,
        unconditional = True,
        skip_connect_scale = None
    ):
        super().__init__()

        # able to upsample video

        self.can_upsample_video = has_temporal_layers

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
            num_conv_kernels = num_conv_kernels,
            style_dims = style_embed_split_dims
        )

        # attention

        full_attn = cast_tuple(full_attn, length = len(dim_mults))
        assert len(full_attn) == len(dim_mults)

        FullAttention = partial(Transformer, flash_attn = flash_attn)

        cross_attn = cast_tuple(cross_attn, length = len(dim_mults))
        assert unconditional or len(full_attn) == len(dim_mults)

        # skip connection scale

        self.skip_connect_scale = default(skip_connect_scale, 2 ** -0.5)

        # layers

        self.downs = ModuleList([])
        self.ups = ModuleList([])
        num_resolutions = len(in_out)
        skip_connect_dims = []

        for ind, ((dim_in, dim_out), layer_full_attn, layer_cross_attn, layer_attn_depth, layer_temporal_attn_depth) in enumerate(zip(in_out, full_attn, cross_attn, attn_depths, temporal_attn_depths)):

            should_not_downsample = ind < num_layer_no_downsample
            has_cross_attn = not self.unconditional and layer_cross_attn

            attn_klass = FullAttention if layer_full_attn else LinearTransformer

            skip_connect_dims.append(dim_in)
            skip_connect_dims.append(dim_in + (dim_out if not should_not_downsample else 0))

            temporal_resnet_block = None
            temporal_attn = None

            if has_temporal_layers:
                temporal_resnet_block = block_klass(dim_in, dim_in, conv_type = '1d')
                temporal_attn = FullAttention(dim_in, dim_head = self_attn_dim_head, heads = self_attn_heads, depth = layer_temporal_attn_depth)

            # all unet downsample stages

            self.downs.append(ModuleList([
                block_klass(dim_in, dim_in),
                block_klass(dim_in, dim_in),
                CrossAttentionBlock(dim_in, dim_context = text_encoder.dim, dim_head = self_attn_dim_head, heads = self_attn_heads, ff_mult = self_attn_ff_mult) if has_cross_attn else None,
                attn_klass(dim_in, dim_head = self_attn_dim_head, heads = self_attn_heads, depth = layer_attn_depth),
                temporal_resnet_block,
                temporal_attn,
                Downsample(dim_in, dim_out, skip_downsample = should_not_downsample, has_temporal_layers = has_temporal_layers)
            ]))

        self.mid_block1 = block_klass(mid_dim, mid_dim)
        self.mid_attn = FullAttention(mid_dim, dim_head = self_attn_dim_head, heads = self_attn_heads, depth = mid_attn_depth)
        self.mid_block2 = block_klass(mid_dim, mid_dim)
        self.mid_to_rgb = nn.Conv2d(mid_dim, channels, 1)

        for ind, ((dim_in, dim_out), layer_cross_attn, layer_full_attn, layer_attn_depth, layer_temporal_attn_depth) in enumerate(zip(reversed(in_out), reversed(full_attn), reversed(cross_attn), reversed(attn_depths), reversed(temporal_attn_depths))):

            attn_klass = FullAttention if layer_full_attn else LinearTransformer
            has_cross_attn = not self.unconditional and layer_cross_attn

            temporal_upsample = None
            temporal_upsample_rgb = None
            temporal_resnet_block = None
            temporal_attn = None

            if has_temporal_layers:
                temporal_upsample = PixelShuffleTemporalUpsample(dim_in, dim_in)
                temporal_upsample_rgb = TemporalUpsample(dim_in, dim_in)

                temporal_resnet_block = block_klass(dim_in, dim_in, conv_type = '1d')
                temporal_attn = FullAttention(dim_in, dim_head = self_attn_dim_head, heads = self_attn_heads, depth = layer_temporal_attn_depth)

            self.ups.append(ModuleList([
                PixelShuffleUpsample(dim_out, dim_in),
                Upsample(),
                temporal_upsample,
                temporal_upsample_rgb,
                nn.Conv2d(dim_in, channels, 1),
                block_klass(dim_in + skip_connect_dims.pop(), dim_in),
                block_klass(dim_in + skip_connect_dims.pop(), dim_in),
                CrossAttentionBlock(dim_in, dim_context = text_encoder.dim, dim_head = self_attn_dim_head, heads = self_attn_heads, ff_mult = cross_ff_mult) if has_cross_attn else None,
                attn_klass(dim_in, dim_head = cross_attn_dim_head, heads = self_attn_heads, depth = layer_attn_depth),
                temporal_resnet_block,
                temporal_attn
            ]))

        self.out_dim = default(out_dim, channels)

        self.final_res_block = block_klass(dim, dim)

        self.final_to_rgb = nn.Conv2d(dim, channels, 1)

        # determine the projection of the style embedding to convolutional modulation weights (+ adaptive kernel selection weights) for all layers

        self.style_to_conv_modulations = nn.Linear(style_network.dim, sum(style_embed_split_dims))
        self.style_embed_split_dims = style_embed_split_dims

    @property
    def allowable_rgb_resolutions(self):
        input_res_base = int(log2(self.input_image_size))
        output_res_base = int(log2(self.image_size))
        allowed_rgb_res_base = list(range(input_res_base, output_res_base))
        return [*map(lambda p: 2 ** p, allowed_rgb_res_base)]

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def total_params(self):
        return sum([p.numel() for p in self.parameters()])

    def resize_to_same_dimensions(self, x, size):
        mode = 'trilinear' if x.ndim == 5 else 'bilinear'
        return F.interpolate(x, tuple(size), mode = mode)

    def forward(
        self,
        lowres_image_or_video,
        styles = None,
        noise = None,
        texts: List[str] | None = None,
        global_text_tokens = None,
        fine_text_tokens = None,
        text_mask = None,
        return_all_rgbs = False,
        replace_rgb_with_input_lowres_image = True   # discriminator should also receive the low resolution image the upsampler sees
    ):
        x = lowres_image_or_video
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

        # first detect whether input is image or video and handle accordingly

        input_is_video = lowres_image_or_video.ndim == 5
        assert not (not self.can_upsample_video and input_is_video), 'this network cannot upsample video unless you set `has_temporal_layers = True`'

        fold_time_into_batch = identity
        split_time_from_batch = identity

        if input_is_video:
            fold_time_into_batch = lambda t: rearrange(t, 'b c t h w -> (b t) c h w')
            split_time_from_batch = lambda t: rearrange(t, '(b t) c h w -> b c t h w', b = batch_size)

        x = fold_time_into_batch(x)

        # set lowres_images for final rgb output

        lowres_images = x

        # initial conv

        x = self.init_conv(x)

        h = []

        # downsample stages

        for (
            block1,
            block2,
            cross_attn,
            attn,
            temporal_block,
            temporal_attn,
            downsample,
        ) in self.downs:

            x = block1(x, conv_mods_iter = conv_mods)
            h.append(x)

            x = block2(x, conv_mods_iter = conv_mods)

            x = attn(x)

            if exists(cross_attn):
                x = cross_attn(x, context = fine_text_tokens, mask = text_mask)

            if input_is_video:
                x = split_time_from_batch(x)
                x, split_space_back = fold_space_into_batch(x)

                x = temporal_block(x, conv_mods_iter = conv_mods)

                x = rearrange(x, 'b c t -> b c t 1')
                x = temporal_attn(x)
                x = rearrange(x, 'b c t 1 -> b c t')

                x = split_space_back(x)
                x = fold_time_into_batch(x)

            elif self.can_upsample_video:
                conv_mods = islice(conv_mods, temporal_block.num_mods, None)

            skip_connect = x

            # downsample with hf shuttle

            x = split_time_from_batch(x)

            x, hf_fmap = downsample(x)

            x = fold_time_into_batch(x)
            hf_fmap = fold_time_into_batch(hf_fmap)

            # add high freq fmap to skip connection as proposed in videogigagan

            skip_connect = torch.cat((skip_connect, hf_fmap), dim = 1)

            h.append(skip_connect)

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

        for (
            upsample,
            upsample_rgb,
            temporal_upsample,
            temporal_upsample_rgb,
            to_rgb,
            block1,
            block2,
            cross_attn,
            attn,
            temporal_block,
            temporal_attn,
        ) in self.ups:

            x = upsample(x)
            rgb = upsample_rgb(rgb)

            if input_is_video:
                x = split_time_from_batch(x)
                rgb = split_time_from_batch(rgb)

                x = temporal_upsample(x)
                rgb = temporal_upsample_rgb(rgb)

                x = fold_time_into_batch(x)
                rgb = fold_time_into_batch(rgb)

            res1 = h.pop() * self.skip_connect_scale
            res2 = h.pop() * self.skip_connect_scale

            # handle skip connections not being the same shape

            if x.shape[0] != res1.shape[0] or x.shape[2:] != res1.shape[2:]:
                x = split_time_from_batch(x)
                res1 = split_time_from_batch(res1)
                res2 = split_time_from_batch(res2)

                res1 = self.resize_to_same_dimensions(res1, x.shape[2:])
                res2 = self.resize_to_same_dimensions(res2, x.shape[2:])

                x = fold_time_into_batch(x)
                res1 = fold_time_into_batch(res1)
                res2 = fold_time_into_batch(res2)

            # concat skip connections

            x = torch.cat((x, res1), dim = 1)
            x = block1(x, conv_mods_iter = conv_mods)

            x = torch.cat((x, res2), dim = 1)
            x = block2(x, conv_mods_iter = conv_mods)

            if exists(cross_attn):
                x = cross_attn(x, context = fine_text_tokens, mask = text_mask)

            x = attn(x)

            if input_is_video:
                x = split_time_from_batch(x)
                x, split_space_back = fold_space_into_batch(x)

                x = temporal_block(x, conv_mods_iter = conv_mods)

                x = rearrange(x, 'b c t -> b c t 1')
                x = temporal_attn(x)
                x = rearrange(x, 'b c t 1 -> b c t')

                x = split_space_back(x)
                x = fold_time_into_batch(x)

            elif self.can_upsample_video:
                conv_mods = islice(conv_mods, temporal_block.num_mods, None)

            rgb = rgb + to_rgb(x)
            rgbs.append(rgb)

        x = self.final_res_block(x, conv_mods_iter = conv_mods)

        assert len([*conv_mods]) == 0

        rgb = rgb + self.final_to_rgb(x)

        # handle video input

        if input_is_video:
            rgb = split_time_from_batch(rgb)

        if not return_all_rgbs:
            return rgb

        # only keep those rgbs whose feature map is greater than the input image to be upsampled

        rgbs = list(filter(lambda t: t.shape[-1] > shape[-1], rgbs))

        # and return the original input image as the smallest rgb

        rgbs = [lowres_images, *rgbs]

        if input_is_video:
            rgbs = [*map(split_time_from_batch, rgbs)]

        return rgb, rgbs

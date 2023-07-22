from collections import namedtuple
from pathlib import Path
from math import log2, sqrt
from functools import partial

from torchvision import utils

import torch
import torch.nn.functional as F
from torch import nn, einsum, Tensor
from torch.autograd import grad as torch_grad
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

from beartype import beartype
from beartype.typing import List, Optional, Tuple, Dict, Union, Iterable

from einops import rearrange, pack, unpack, repeat, reduce
from einops.layers.torch import Rearrange, Reduce

from kornia.filters import filter2d

from ema_pytorch import EMA

from gigagan_pytorch.version import __version__
from gigagan_pytorch.open_clip import OpenClipAdapter
from gigagan_pytorch.optimizer import get_optimizer

from tqdm import tqdm

from numerize import numerize

from accelerate import Accelerator, DistributedType
from accelerate.utils import DistributedDataParallelKwargs

# helpers

def exists(val):
    return val is not None

@beartype
def is_empty(arr: Union[Tuple, Dict, List]):
    return len(arr) == 0

def default(*vals):
    for val in vals:
        if exists(val):
            return val
    return None

def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

def is_power_of_two(n):
    return log2(n).is_integer()

def safe_unshift(arr):
    if len(arr) == 0:
        return None
    return arr.pop(0)

def divisible_by(numer, denom):
    return (numer % denom) == 0

def group_by_num_consecutive(arr, num):
    out = []
    for ind, el in enumerate(arr):
        if ind > 0 and divisible_by(ind, num):
            yield out
            out = []

        out.append(el)

    if len(out) > 0:
        yield out

def is_unique(arr):
    return len(set(arr)) == len(arr)

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups, remainder = divmod(num, divisor)
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def mkdir_if_not_exists(path):
    path.mkdir(exist_ok = True, parents = True)

@beartype
def set_requires_grad_(
    m: nn.Module,
    requires_grad: bool
):
    for p in m.parameters():
        p.requires_grad = requires_grad

# activation functions

def leaky_relu(neg_slope = 0.1):
    return nn.LeakyReLU(neg_slope)

def conv2d_3x3(dim_in, dim_out):
    return nn.Conv2d(dim_in, dim_out, 3, padding = 1)

# tensor helpers

def log(t, eps = 1e-20):
    return t.clamp(min = eps).log()

def gradient_penalty(
    images,
    outputs,
    grad_output_weights = None,
    weight = 10,
    scaler: Optional[GradScaler] = None
):
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]

    if exists(scaler):
        outputs = [*map(scaler.scale, outputs)]

    if not exists(grad_output_weights):
        grad_output_weights = (1,) * len(outputs)

    maybe_scaled_gradients, *_ = torch_grad(
        outputs = outputs,
        inputs = images,
        grad_outputs = [(torch.ones_like(output) * weight) for output, weight in zip(outputs, grad_output_weights)],
        create_graph = True,
        retain_graph = True,
        only_inputs = True
    )

    gradients = maybe_scaled_gradients

    if exists(scaler):
        scale = scaler.get_scale()
        inv_scale = 1. / max(scale, 1e-6)
        gradients = maybe_scaled_gradients * inv_scale

    gradients = rearrange(gradients, 'b ... -> b (...)')
    return weight * ((gradients.norm(2, dim = 1) - 1) ** 2).mean()

# hinge gan losses

def generator_hinge_loss(fake):
    return fake.mean()

def discriminator_hinge_loss(real, fake):
    return (F.relu(1 + real) + F.relu(1 - fake)).mean()

# auxiliary losses

def aux_matching_loss(real, fake):
    """
    making logits negative, as in this framework, discriminator is 0 for real, high value for fake. GANs can have this arbitrarily swapped, as it only matters if the generator and discriminator are opposites
    """
    return log(1 + (-real).exp()) + log(1 + (-fake).exp())

@beartype
def aux_clip_loss(
    clip: OpenClipAdapter,
    images: Tensor,
    texts: Optional[List[str]] = None,
    text_embeds: Optional[Tensor] = None
):
    assert exists(texts) ^ exists(text_embeds)

    if exists(texts):
        text_embeds, _ = clip.embed_texts(texts)

    return clip.contrastive_loss(images = images, text_embeds = text_embeds)

# rmsnorm (newer papers show mean-centering in layernorm not necessary)

class ChannelRMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        normed = F.normalize(x, dim = 1)
        return normed * self.scale * self.gamma

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        normed = F.normalize(x, dim = -1)
        return normed * self.scale * self.gamma

# down and upsample

class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)

    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f [None, :, None]
        return filter2d(x, f, normalized = True)

def Upsample(*args):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = False),
        Blur()
    )

class PixelShuffleUpsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        conv = nn.Conv2d(dim, dim * 4, 1)

        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            nn.PixelShuffle(2)
        )

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, h, w = conv.weight.shape
        conv_weight = torch.empty(o // 4, i, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o 4) ...')

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        return self.net(x)

def Downsample(dim):
    return nn.Sequential(
        Rearrange('b c (h s1) (w s2) -> b (c s1 s2) h w', s1 = 2, s2 = 2),
        nn.Conv2d(dim * 4, dim, 1)
    )

# skip layer excitation

def SqueezeExcite(dim, dim_out, reduction = 4, dim_min = 32):
    dim_hidden = max(dim_out // reduction, dim_min)

    return nn.Sequential(
        Reduce('b c h w -> b c', 'mean'),
        nn.Linear(dim, dim_hidden),
        nn.SiLU(),
        nn.Linear(dim_hidden, dim_out),
        nn.Sigmoid(),
        Rearrange('b c -> b c 1 1')
    )

# adaptive conv
# the main novelty of the paper - they propose to learn a softmax weighted sum of N convolutional kernels, depending on the text embedding

def get_same_padding(size, kernel, dilation, stride):
    return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

class AdaptiveConv2DMod(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        kernel,
        *,
        demod = True,
        stride = 1,
        dilation = 1,
        eps = 1e-8,
        num_conv_kernels = 1 # set this to be greater than 1 for adaptive
    ):
        super().__init__()
        self.eps = eps

        self.dim_out = dim_out

        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.adaptive = num_conv_kernels > 1

        self.weights = nn.Parameter(torch.randn((num_conv_kernels, dim_out, dim, kernel, kernel)))

        self.demod = demod

        nn.init.kaiming_normal_(self.weights, a = 0, mode = 'fan_in', nonlinearity = 'leaky_relu')

    def forward(
        self,
        fmap,
        mod: Optional[Tensor] = None,
        kernel_mod: Optional[Tensor] = None
    ):
        """
        notation

        b - batch
        n - convs
        o - output
        i - input
        k - kernel
        """

        b, h = fmap.shape[0], fmap.shape[-2]

        # account for feature map that has been expanded by the scale in the first dimension
        # due to multiscale inputs and outputs

        if mod.shape[0] != b:
            mod = repeat(mod, 'b ... -> (s b) ...', s = b // mod.shape[0])

        if kernel_mod.shape[0] != b:
            kernel_mod = repeat(kernel_mod, 'b ... -> (s b) ...', s = b // kernel_mod.shape[0])

        # prepare weights for modulation

        weights = self.weights

        if self.adaptive:
            weights = repeat(weights, '... -> b ...', b = b)

            # determine an adaptive weight and 'select' the kernel to use with softmax

            assert exists(kernel_mod)

            kernel_attn = kernel_mod.softmax(dim = -1)
            kernel_attn = rearrange(kernel_attn, 'b n -> b n 1 1 1 1')

            weights = reduce(weights * kernel_attn, 'b n ... -> b ...', 'sum')

        # do the modulation, demodulation, as done in stylegan2

        mod = rearrange(mod, 'b i -> b 1 i 1 1')

        weights = weights * (mod + 1)

        if self.demod:
            inv_norm = reduce(weights ** 2, 'b o i k1 k2 -> b o 1 1 1', 'sum').clamp(min = self.eps).rsqrt()
            weights = weights * inv_norm

        fmap = rearrange(fmap, 'b c h w -> 1 (b c) h w')

        weights = rearrange(weights, 'b o ... -> (b o) ...')

        padding = get_same_padding(h, self.kernel, self.dilation, self.stride)
        fmap = F.conv2d(fmap, weights, padding = padding, groups = b)

        return rearrange(fmap, '1 (b o) ... -> b o ...', b = b)

# attention
# they use an attention with a better Lipchitz constant - l2 distance similarity instead of dot product - also shared query / key space - shown in vitgan to be more stable
# not sure what they did about token attention to self, so masking out, as done in some other papers using shared query / key space

class SelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        dot_product = False
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads

        self.dot_product = dot_product

        self.norm = ChannelRMSNorm(dim)

        self.to_q = nn.Conv2d(dim, dim_inner, 1, bias = False)
        self.to_k = nn.Conv2d(dim, dim_inner, 1, bias = False) if dot_product else None
        self.to_v = nn.Conv2d(dim, dim_inner, 1, bias = False)

        self.null_kv = nn.Parameter(torch.randn(2, heads, dim_head))

        self.to_out = nn.Conv2d(dim_inner, dim, 1, bias = False)

    def forward(self, fmap):
        """
        einstein notation

        b - batch
        h - heads
        x - height
        y - width
        d - dimension
        i - source seq (attend from)
        j - target seq (attend to)
        """
        batch = fmap.shape[0]

        fmap = self.norm(fmap)

        x, y = fmap.shape[-2:]

        h = self.heads

        q, v = self.to_q(fmap), self.to_v(fmap)

        k = self.to_k(fmap) if exists(self.to_k) else q

        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> (b h) (x y) d', h = self.heads), (q, k, v))

        # add a null key / value, so network can choose to pay attention to nothing

        nk, nv = map(lambda t: repeat(t, 'h d -> (b h) 1 d', b = batch), self.null_kv)

        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        # l2 distance or dot product

        if self.dot_product:
            sim = einsum('b i d, b j d -> b i j', q, k)
        else:
            # using pytorch cdist leads to nans in lightweight gan training framework, at least
            q_squared = (q * q).sum(dim = -1)
            k_squared = (k * k).sum(dim = -1)
            l2dist_squared = rearrange(q_squared, 'b i -> b i 1') + rearrange(k_squared, 'b j -> b 1 j') - 2 * einsum('b i d, b j d -> b i j', q, k) # hope i'm mathing right
            sim = -l2dist_squared

        # scale

        sim = sim * self.scale

        # attention

        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)

        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', x = x, y = y, h = h)

        return self.to_out(out)

class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_context,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads
        kv_input_dim = default(dim_context, dim)

        self.norm = ChannelRMSNorm(dim)
        self.norm_context = RMSNorm(kv_input_dim)

        self.to_q = nn.Conv2d(dim, dim_inner, 1, bias = False)
        self.to_kv = nn.Linear(kv_input_dim, dim_inner * 2, bias = False)
        self.to_out = nn.Conv2d(dim_inner, dim, 1, bias = False)

    def forward(self, fmap, context, mask = None):
        """
        einstein notation

        b - batch
        h - heads
        x - height
        y - width
        d - dimension
        i - source seq (attend from)
        j - target seq (attend to)
        """

        fmap = self.norm(fmap)
        context = self.norm_context(context)

        x, y = fmap.shape[-2:]

        h = self.heads

        q, k, v = (self.to_q(fmap), *self.to_kv(context).chunk(2, dim = -1))

        k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (k, v))

        q = rearrange(q, 'b (h d) x y -> (b h) (x y) d', h = self.heads)

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = repeat(mask, 'b j -> (b h) 1 j', h = self.heads)
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)

        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', x = x, y = y, h = h)

        return self.to_out(out)

# classic transformer attention, stick with l2 distance

class TextAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias = False)

        self.null_kv = nn.Parameter(torch.randn(2, heads, dim_head))

        self.to_out = nn.Linear(dim_inner, dim, bias = False)

    def forward(self, encodings, mask = None):
        """
        einstein notation

        b - batch
        h - heads
        x - height
        y - width
        d - dimension
        i - source seq (attend from)
        j - target seq (attend to)
        """
        batch = encodings.shape[0]

        encodings = self.norm(encodings)

        h = self.heads

        q, k, v = self.to_qkv(encodings).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = self.heads), (q, k, v))

        # add a null key / value, so network can choose to pay attention to nothing

        nk, nv = map(lambda t: repeat(t, 'h d -> (b h) 1 d', b = batch), self.null_kv)

        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        # key padding mask

        if exists(mask):
            mask = F.pad(mask, (1, 0), value = True)
            mask = repeat(mask, 'b n -> (b h) 1 n', h = h)
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # attention

        attn = sim.softmax(dim = -1)
        out = einsum('b i j, b j d -> b i d', attn, v)

        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)

        return self.to_out(out)

# feedforward

def FeedForward(
    dim,
    mult = 4,
    channel_first = False
):
    dim_hidden = int(dim * mult)
    norm_klass = ChannelRMSNorm if channel_first else RMSNorm
    proj = partial(nn.Conv2d, kernel_size = 1) if channel_first else nn.Linear

    return nn.Sequential(
        norm_klass(dim),
        proj(dim, dim_hidden),
        nn.GELU(),
        proj(dim_hidden, dim)
    )

# different types of transformer blocks or transformers (multiple blocks)

class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        dot_product = False
    ):
        super().__init__()
        self.attn = SelfAttention(dim = dim, dim_head = dim_head, heads = heads, dot_product = dot_product)
        self.ff = FeedForward(dim = dim, mult = ff_mult, channel_first = True)

    def forward(self, x):
        x = self.attn(x) + x
        x = self.ff(x) + x
        return x

class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_context,
        dim_head = 64,
        heads = 8,
        ff_mult = 4
    ):
        super().__init__()
        self.attn = CrossAttention(dim = dim, dim_context = dim_context, dim_head = dim_head, heads = heads)
        self.ff = FeedForward(dim = dim, mult = ff_mult, channel_first = True)

    def forward(self, x, context, mask = None):
        x = self.attn(x, context = context, mask = mask) + x
        x = self.ff(x) + x
        return x

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                TextAttention(dim = dim, dim_head = dim_head, heads = heads),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        self.norm = RMSNorm(dim)

    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask) + x
            x = ff(x) + x

        return self.norm(x)

# text encoder

class TextEncoder(nn.Module):
    @beartype
    def __init__(
        self,
        *,
        dim,
        depth,
        clip: Optional[OpenClipAdapter] = None,
        dim_head = 64,
        heads = 8,
    ):
        super().__init__()
        self.dim = dim

        if not exists(clip):
            clip = OpenClipAdapter()

        self.clip = clip
        set_requires_grad_(clip, False)

        self.learned_global_token = nn.Parameter(torch.randn(dim))

        self.project_in = nn.Linear(clip.dim_latent, dim) if clip.dim_latent != dim else nn.Identity()

        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            dim_head = dim_head,
            heads = heads
        )

    @beartype
    def forward(
        self,
        texts: Optional[List[str]] = None,
        text_encodings: Optional[Tensor] = None
    ):
        assert exists(texts) ^ exists(text_encodings)

        if not exists(text_encodings):
            with torch.no_grad():
                self.clip.eval()
                _, text_encodings = self.clip.embed_texts(texts)

        mask = (text_encodings != 0.).any(dim = -1)

        text_encodings = self.project_in(text_encodings)

        mask_with_global = F.pad(mask, (1, 0), value = True)

        batch = text_encodings.shape[0]
        global_tokens = repeat(self.learned_global_token, 'd -> b d', b = batch)

        text_encodings, ps = pack([global_tokens, text_encodings], 'b * d')

        text_encodings = self.transformer(text_encodings, mask = mask_with_global)

        global_tokens, text_encodings = unpack(text_encodings, ps, 'b * d')

        return global_tokens, text_encodings, mask

# style mapping network

class EqualLinear(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        lr_mul = 1,
        bias = True
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim_out, dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(dim_out))

        self.lr_mul = lr_mul

    def forward(self, input):
        return F.linear(input, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)

class StyleNetwork(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        lr_mul = 0.1,
        dim_text_latent = 0
    ):
        super().__init__()
        self.dim = dim
        self.dim_text_latent = dim_text_latent

        layers = []
        for i in range(depth):
            is_first = i == 0
            dim_in = (dim + dim_text_latent) if is_first else dim

            layers.extend([EqualLinear(dim_in, dim, lr_mul), leaky_relu()])

        self.net = nn.Sequential(*layers)

    def forward(
        self,
        x,
        text_latent = None
    ):
        x = F.normalize(x, dim = 1)

        if self.dim_text_latent > 0:
            assert exists(text_latent)
            x = torch.cat((x, text_latent), dim = -1)

        return self.net(x)

# noise

class Noise(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim, 1, 1))

    def forward(
        self,
        x,
        noise = None
    ):
        b, _, h, w, device = *x.shape, x.device

        if not exists(noise):
            noise = torch.randn(b, 1, h, w, device = device)

        return x + self.weight * noise

# generator

class BaseGenerator(nn.Module):
    pass

class Generator(BaseGenerator):
    @beartype
    def __init__(
        self,
        *,
        image_size,
        dim_capacity = 16,
        dim_max = 2048,
        channels = 3,
        style_network: Optional[Union[StyleNetwork, Dict]] = None,
        style_network_dim = None,
        text_encoder: Optional[Union[TextEncoder, Dict]] = None,
        dim_latent = 512,
        self_attn_resolutions: Tuple[int, ...] = (32, 16),
        self_attn_dim_head = 64,
        self_attn_heads = 8,
        self_attn_dot_product = True,
        self_attn_ff_mult = 4,
        cross_attn_resolutions: Tuple[int, ...] = (32, 16),
        cross_attn_dim_head = 64,
        cross_attn_heads = 8,
        cross_attn_ff_mult = 4,
        num_conv_kernels = 2,  # the number of adaptive conv kernels
        num_skip_layers_excite = 0,
        unconditional = False,
        use_glu = True,
        pixel_shuffle_upsample = False
    ):
        super().__init__()
        self.channels = channels

        if isinstance(style_network, dict):
            style_network = StyleNetwork(**style_network)

        self.style_network = style_network

        assert exists(style_network) ^ exists(style_network_dim), 'style_network_dim must be given to the generator if StyleNetwork not passed in as style_network'

        if not exists(style_network_dim):
            style_network_dim = style_network.dim

        self.style_network_dim = style_network_dim

        if isinstance(text_encoder, dict):
            text_encoder = TextEncoder(**text_encoder)

        self.text_encoder = text_encoder

        self.unconditional = unconditional

        assert not (unconditional and exists(text_encoder))
        assert not (unconditional and exists(style_network) and style_network.dim_text_latent > 0)
        assert unconditional or (exists(text_encoder) and text_encoder.dim == style_network.dim_text_latent), 'the `dim_text_latent` on your StyleNetwork must be equal to the `dim` set for the TextEncoder'

        assert is_power_of_two(image_size)
        num_layers = int(log2(image_size) - 1)
        self.num_layers = num_layers

        # generator requires convolutions conditioned by the style vector
        # and also has N convolutional kernels adaptively selected (one of the only novelties of the paper)

        is_adaptive = num_conv_kernels > 1
        dim_kernel_mod = num_conv_kernels if is_adaptive else 0

        style_embed_split_dims = []

        adaptive_conv = partial(AdaptiveConv2DMod, kernel = 3, num_conv_kernels = num_conv_kernels)

        # initial 4x4 block and conv

        self.init_block = nn.Parameter(torch.randn(dim_latent, 4, 4))
        self.init_conv = adaptive_conv(dim_latent, dim_latent)

        style_embed_split_dims.extend([
            dim_latent,
            dim_kernel_mod
        ])

        # main network

        num_layers = int(log2(image_size) - 1)
        self.num_layers = num_layers

        resolutions = image_size / ((2 ** torch.arange(num_layers)))
        resolutions = resolutions.long().tolist()

        dim_layers = (2 ** (torch.arange(num_layers) + 1)) * dim_capacity
        dim_layers.clamp_(max = dim_max)

        dim_layers = torch.flip(dim_layers, (0,))
        dim_layers = F.pad(dim_layers, (1, 0), value = dim_latent)

        dim_layers = dim_layers.tolist()

        dim_pairs = list(zip(dim_layers[:-1], dim_layers[1:]))

        self.num_skip_layers_excite = num_skip_layers_excite

        self.layers = nn.ModuleList([])

        # go through layers and construct all parameters

        for ind, ((dim_in, dim_out), resolution) in enumerate(zip(dim_pairs, resolutions)):
            is_last = (ind + 1) == len(dim_pairs)
            is_first = ind == 0

            should_upsample = not is_first
            should_upsample_rgb = not is_last
            should_skip_layer_excite = num_skip_layers_excite > 0 and (ind + num_skip_layers_excite) < len(dim_pairs)

            has_self_attn = resolution in self_attn_resolutions
            has_cross_attn = resolution in cross_attn_resolutions and not unconditional

            skip_squeeze_excite = None
            if should_skip_layer_excite:
                dim_skip_in, _ = dim_pairs[ind + num_skip_layers_excite]
                skip_squeeze_excite = SqueezeExcite(dim_in, dim_skip_in)

            dim_inner = dim_out * (2 if use_glu else 1)
            activation = partial(nn.GLU, dim = 1) if use_glu else leaky_relu

            resnet_block = nn.ModuleList([
                adaptive_conv(dim_in, dim_inner),
                Noise(dim_inner),
                activation(),
                adaptive_conv(dim_out, dim_inner),
                Noise(dim_inner),
                activation()
            ])

            to_rgb = adaptive_conv(dim_out, channels)

            self_attn = cross_attn = rgb_upsample = upsample = None

            upsample_klass = Upsample if not pixel_shuffle_upsample else PixelShuffleUpsample

            upsample = upsample_klass(dim_in) if should_upsample else None
            rgb_upsample = upsample_klass(channels) if should_upsample_rgb else None

            if has_self_attn:
                self_attn = SelfAttentionBlock(
                    dim_out,
                    dim_head = self_attn_dim_head,
                    heads = self_attn_heads,
                    ff_mult = self_attn_ff_mult,
                    dot_product = self_attn_dot_product
            )

            if has_cross_attn:
                cross_attn = CrossAttentionBlock(
                    dim_out,
                    dim_context = text_encoder.dim,
                    dim_head = cross_attn_dim_head,
                    heads = cross_attn_heads,
                    ff_mult = cross_attn_ff_mult,
                )

            style_embed_split_dims.extend([
                dim_in,             # for first conv in resnet block
                dim_kernel_mod,     # first conv kernel selection
                dim_out,            # second conv in resnet block
                dim_kernel_mod,     # second conv kernel selection
                dim_out,            # to RGB conv
                dim_kernel_mod,     # RGB conv kernel selection
            ])

            self.layers.append(nn.ModuleList([
                skip_squeeze_excite,
                resnet_block,
                to_rgb,
                self_attn,
                cross_attn,
                upsample,
                rgb_upsample
            ]))

        # determine the projection of the style embedding to convolutional modulation weights (+ adaptive kernel selection weights) for all layers

        self.style_to_conv_modulations = nn.Linear(style_network_dim, sum(style_embed_split_dims))
        self.style_embed_split_dims = style_embed_split_dims

        self.apply(self.init_)
        nn.init.normal_(self.init_block, std = 0.02)

    def init_(self, m):
        if type(m) in {nn.Conv2d, nn.Linear}:
            nn.init.kaiming_normal_(m.weight, a = 0, mode = 'fan_in', nonlinearity = 'leaky_relu')

    @property
    def total_params(self):
        return sum([p.numel() for p in self.parameters() if p.requires_grad])

    @property
    def device(self):
        return next(self.parameters()).device

    @beartype
    def forward(
        self,
        styles = None,
        noise = None,
        texts: Optional[List[str]] = None,
        text_encodings: Optional[Tensor] = None,
        global_text_tokens = None,
        fine_text_tokens = None,
        text_mask = None,
        batch_size = 1,
        return_all_rgbs = False
    ):
        # take care of text encodings
        # which requires global text tokens to adaptively select the kernels from the main contribution in the paper
        # and fine text tokens to attend to using cross attention

        if not self.unconditional:
            if exists(texts) or exists(text_encodings):
                assert exists(texts) ^ exists(text_encodings), 'either raw texts as List[str] or text_encodings (from clip) as Tensor is passed in, but not both'
                assert exists(self.text_encoder)

                if exists(texts):
                    text_encoder_kwargs = dict(texts = texts)
                elif exists(text_encodings):
                    text_encoder_kwargs = dict(text_encodings = text_encodings)

                global_text_tokens, fine_text_tokens, text_mask = self.text_encoder(**text_encoder_kwargs)
            else:
                assert all([*map(exists, (global_text_tokens, fine_text_tokens, text_mask))]), 'raw text or text embeddings were not passed in for conditional training'
        else:
            assert not any([*map(exists, (texts, global_text_tokens, fine_text_tokens))])

        # determine styles

        if not exists(styles):
            assert exists(self.style_network)

            if not exists(noise):
                noise = torch.randn((batch_size, self.style_network_dim), device = self.device)

            styles = self.style_network(noise, global_text_tokens)

        # project styles to conv modulations

        conv_mods = self.style_to_conv_modulations(styles)
        conv_mods = conv_mods.split(self.style_embed_split_dims, dim = -1)
        conv_mods = iter(conv_mods)

        # prepare initial block

        batch_size = styles.shape[0]

        x = repeat(self.init_block, 'c h w -> b c h w', b = batch_size)
        x = self.init_conv(x, mod = next(conv_mods), kernel_mod = next(conv_mods))

        rgb = torch.zeros((batch_size, self.channels, 4, 4), device = self.device, dtype = x.dtype)

        # skip layer squeeze excitations

        excitations = [None] * self.num_skip_layers_excite

        # all the rgb's of each layer of the generator is to be saved for multi-resolution input discrimination

        rgbs = []

        # main network

        for squeeze_excite, (resnet_conv1, noise1, act1, resnet_conv2, noise2, act2), to_rgb_conv, self_attn, cross_attn, upsample, upsample_rgb in self.layers:

            if exists(upsample):
                x = upsample(x)

            if exists(squeeze_excite):
                skip_excite = squeeze_excite(x)
                excitations.append(skip_excite)

            excite = safe_unshift(excitations)
            if exists(excite):
                x = x * excite

            x = resnet_conv1(x, mod = next(conv_mods), kernel_mod = next(conv_mods))
            x = noise1(x)
            x = act1(x)

            x = resnet_conv2(x, mod = next(conv_mods), kernel_mod = next(conv_mods))
            x = noise2(x)
            x = act2(x)

            if exists(self_attn):
                x = self_attn(x)

            if exists(cross_attn):
                x = cross_attn(x, context = fine_text_tokens, mask = text_mask)

            layer_rgb = to_rgb_conv(x, mod = next(conv_mods), kernel_mod = next(conv_mods))

            rgb = rgb + layer_rgb

            rgbs.append(rgb)

            if exists(upsample_rgb):
                rgb = upsample_rgb(rgb)

        # sanity check

        assert is_empty([*conv_mods]), 'convolutions were incorrectly modulated'

        if return_all_rgbs:
            return rgb, rgbs

        return rgb

# discriminator

@beartype
class SimpleDecoder(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dims: Tuple[int, ...],
        patch_dim: int = 1,
        frac_patches: float = 1.,
        dropout: float = 0.5
    ):
        super().__init__()
        assert 0 < frac_patches <= 1.

        self.patch_dim = patch_dim
        self.frac_patches = frac_patches

        self.dropout = nn.Dropout(dropout)

        dims = [dim, *dims]

        layers = [conv2d_3x3(dim, dim)]

        for dim_in, dim_out in zip(dims[:-1], dims[1:]):
            layers.append(nn.Sequential(
                Upsample(dim_in),
                conv2d_3x3(dim_in, dim_out),
                leaky_relu()
            ))

        self.net = nn.Sequential(*layers)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        fmap,
        orig_image
    ):
        fmap = self.dropout(fmap)

        if self.frac_patches < 1.:
            batch, patch_dim = fmap.shape[0], self.patch_dim
            fmap_size, img_size = fmap.shape[-1], orig_image.shape[-1]

            assert divisible_by(fmap_size, patch_dim), f'feature map dimensions are {fmap_size}, but the patch dim was designated to be {patch_dim}'
            assert divisible_by(img_size, patch_dim), f'image size is {img_size} but the patch dim was specified to be {patch_dim}'

            fmap, orig_image = map(lambda t: rearrange(t, 'b c (p1 h) (p2 w) -> b (p1 p2) c h w', p1 = patch_dim, p2 = patch_dim), (fmap, orig_image))

            total_patches = patch_dim ** 2
            num_patches_recon = max(int(self.frac_patches * total_patches), 1)

            batch_arange = torch.arange(batch, device = self.device)[..., None]
            batch_randperm = torch.randn((batch, total_patches)).sort(dim = -1).indices
            patch_indices = batch_randperm[..., :num_patches_recon]

            fmap, orig_image = map(lambda t: t[batch_arange, patch_indices], (fmap, orig_image))
            fmap, orig_image = map(lambda t: rearrange(t, 'b p ... -> (b p) ...'), (fmap, orig_image))

        recon = self.net(fmap)
        return F.mse_loss(recon, orig_image)

class RandomFixedProjection(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        channel_first = True
    ):
        super().__init__()
        weights = torch.randn(dim, dim_out)
        nn.init.kaiming_normal_(weights, mode = 'fan_out', nonlinearity = 'linear')

        self.channel_first = channel_first
        self.register_buffer('fixed_weights', weights)

    def forward(self, x):
        if not self.channel_first:
            return x @ self.fixed_weights

        return einsum('b c ..., c d -> b d ...', x, self.fixed_weights)

class VisionAidedDiscriminator(nn.Module):
    """ the vision-aided gan loss """

    @beartype
    def __init__(
        self,
        *,
        depth = 2,
        dim_head = 64,
        heads = 8,
        clip: Optional[OpenClipAdapter] = None,
        layer_indices = (-1, -2, -3),
        conv_dim = None,
        text_dim = None,
        unconditional = False,
        num_conv_kernels = 2
    ):
        super().__init__()

        if not exists(clip):
            clip = OpenClipAdapter()

        set_requires_grad_(clip, False)

        self.clip = clip
        dim = clip._dim_image_latent

        self.unconditional = unconditional
        text_dim = default(text_dim, dim)
        conv_dim = default(conv_dim, dim)

        self.layer_discriminators = nn.ModuleList([])
        self.layer_indices = layer_indices

        conv_klass = partial(AdaptiveConv2DMod, kernel = 3, num_conv_kernels = num_conv_kernels) if not unconditional else conv2d_3x3

        for _ in layer_indices:
            self.layer_discriminators.append(nn.ModuleList([
                RandomFixedProjection(dim, conv_dim),
                conv_klass(conv_dim, conv_dim),
                nn.Linear(text_dim, conv_dim) if not unconditional else None,
                nn.Linear(text_dim, num_conv_kernels) if not unconditional else None,
                nn.Sequential(
                    conv2d_3x3(conv_dim, 1),
                    Rearrange('b 1 ... -> b ...')
                )
            ]))

    @property
    def total_params(self):
        return sum([p.numel() for p in self.parameters() if p.requires_grad])

    @beartype
    def forward(
        self,
        images,
        texts: Optional[List[str]] = None,
        text_embeds: Optional[Tensor] = None
    ) -> List[Tensor]:
        assert self.unconditional or (exists(text_embeds) ^ exists(texts))

        with torch.no_grad():
            if not self.unconditional and exists(texts):
                self.clip.eval()
                text_embeds = self.clip.embed_texts

            self.clip.eval()
            _, image_encodings = self.clip.embed_images(images)
            image_encodings = image_encodings.detach()

        logits = []
        for layer_index, (rand_proj, conv, to_conv_mod, to_conv_kernel_mod, to_logits) in zip(self.layer_indices, self.layer_discriminators):
            image_encoding = image_encodings[layer_index]

            cls_token, rest_tokens = image_encoding[:, :1], image_encoding[:, 1:]
            height_width = int(sqrt(rest_tokens.shape[-2])) # assume square

            img_fmap = rearrange(rest_tokens, 'b (h w) d -> b d h w', h = height_width)

            img_fmap = img_fmap + rearrange(cls_token, 'b 1 d -> b d 1 1 ') # pool the cls token into the rest of the tokens
            img_fmap = rand_proj(img_fmap)

            if self.unconditional:
                img_fmap = conv(img_fmap)
            else:
                assert exists(text_embeds)

                img_fmap = conv(
                    img_fmap,
                    mod = to_conv_mod(text_embeds),
                    kernel_mod = to_conv_kernel_mod(text_embeds)
                )

            layer_logits = to_logits(img_fmap)

            logits.append(layer_logits)

        return logits

class Predictor(nn.Module):
    def __init__(
        self,
        dim,
        depth = 4,
        num_conv_kernels = 2,
        unconditional = False
    ):
        super().__init__()
        self.unconditional = unconditional
        self.residual_fn = nn.Conv2d(dim, dim, 1)
        self.residual_scale = 2 ** -0.5

        self.layers = nn.ModuleList([])

        klass = nn.Conv2d if unconditional else partial(AdaptiveConv2DMod, num_conv_kernels = num_conv_kernels)
        klass_kwargs = dict(padding = 1) if unconditional else dict()

        for ind in range(depth):
            self.layers.append(nn.ModuleList([
                klass(dim, dim, 3, **klass_kwargs),
                leaky_relu(),
                klass(dim, dim, 3, **klass_kwargs),
                leaky_relu()
            ]))

        self.to_logits = nn.Conv2d(dim, 1, 1)

    def forward(
        self,
        x,
        mod = None,
        kernel_mod = None
    ):
        residual = self.residual_fn(x)

        kwargs = dict()

        if not self.unconditional:
            kwargs = dict(mod = mod, kernel_mod = kernel_mod)

        for conv1, activation, conv2, activation in self.layers:

            inner_residual = x

            x = conv1(x, **kwargs)
            x = activation(x)
            x = conv2(x, **kwargs)
            x = activation(x)

            x = x + inner_residual
            x = x * self.residual_scale

        x = x + residual
        return self.to_logits(x)

class Discriminator(nn.Module):
    @beartype
    def __init__(
        self,
        *,
        dim_capacity = 16,
        image_size,
        dim_max = 2048,
        channels = 3,
        attn_resolutions: Tuple[int, ...] = (32, 16),
        attn_dim_head = 64,
        attn_heads = 8,
        self_attn_dot_product = False,
        ff_mult = 4,
        text_encoder: Optional[Union[TextEncoder, Dict]] = None,
        text_dim = None,
        filter_input_resolutions: bool = True,
        multiscale_input_resolutions: Tuple[int, ...] = (64, 32, 16, 8),
        multiscale_output_skip_stages: int = 1,
        aux_recon_resolutions: Tuple[int, ...] = (8,),
        aux_recon_patch_dims: Tuple[int, ...] = (2,),
        aux_recon_frac_patches: Tuple[float, ...] = (0.25,),
        aux_recon_fmap_dropout: float = 0.5,
        resize_mode = 'bilinear',
        num_conv_kernels = 2,
        num_skip_layers_excite = 0,
        unconditional = False,
        predictor_depth = 2
    ):
        super().__init__()
        self.unconditional = unconditional
        assert not (unconditional and exists(text_encoder))

        assert is_power_of_two(image_size)
        assert all([*map(is_power_of_two, attn_resolutions)])

        if filter_input_resolutions:
            multiscale_input_resolutions = [*filter(lambda t: t < image_size, multiscale_input_resolutions)]

        assert is_unique(multiscale_input_resolutions)
        assert all([*map(is_power_of_two, multiscale_input_resolutions)])
        assert all([*map(lambda t: t < image_size, multiscale_input_resolutions)])

        self.multiscale_input_resolutions = multiscale_input_resolutions

        assert multiscale_output_skip_stages > 0
        multiscale_output_resolutions = [resolution // (2 ** multiscale_output_skip_stages) for resolution in multiscale_input_resolutions]

        assert all([*map(lambda t: t >= 4, multiscale_output_resolutions)])

        assert all([*map(lambda t: t < image_size, multiscale_output_resolutions)])

        if len(multiscale_input_resolutions) > 0 and len(multiscale_output_resolutions) > 0:
            assert max(multiscale_input_resolutions) > max(multiscale_output_resolutions)
            assert min(multiscale_input_resolutions) > min(multiscale_output_resolutions)

        self.multiscale_output_resolutions = multiscale_output_resolutions

        assert all([*map(is_power_of_two, aux_recon_resolutions)])
        assert len(aux_recon_resolutions) == len(aux_recon_patch_dims) == len(aux_recon_frac_patches)

        self.aux_recon_resolutions_to_patches = {resolution: (patch_dim, frac_patches) for resolution, patch_dim, frac_patches in zip(aux_recon_resolutions, aux_recon_patch_dims, aux_recon_frac_patches)}

        self.resize_mode = resize_mode

        num_layers = int(log2(image_size) - 1)
        self.num_layers = num_layers
        self.image_size = image_size

        resolutions = image_size / ((2 ** torch.arange(num_layers)))
        resolutions = resolutions.long().tolist()

        dim_layers = (2 ** (torch.arange(num_layers) + 1)) * dim_capacity
        dim_layers = F.pad(dim_layers, (1, 0), value = channels)
        dim_layers.clamp_(max = dim_max)

        dim_layers = dim_layers.tolist()
        dim_last = dim_layers[-1]
        dim_pairs = list(zip(dim_layers[:-1], dim_layers[1:]))

        self.num_skip_layers_excite = num_skip_layers_excite

        self.residual_scale = 2 ** -0.5
        self.layers = nn.ModuleList([])

        upsample_dims = []
        predictor_dims = []
        dim_kernel_attn = (num_conv_kernels if num_conv_kernels > 1 else 0)

        for ind, ((dim_in, dim_out), resolution) in enumerate(zip(dim_pairs, resolutions)):
            is_first = ind == 0
            is_last = (ind + 1) == len(dim_pairs)
            should_downsample = not is_last
            should_skip_layer_excite = not is_first and num_skip_layers_excite > 0 and (ind + num_skip_layers_excite) < len(dim_pairs)

            has_attn = resolution in attn_resolutions
            has_multiscale_output = resolution in multiscale_output_resolutions

            has_aux_recon_decoder = resolution in aux_recon_resolutions
            upsample_dims.insert(0, dim_in)

            skip_squeeze_excite = None
            if should_skip_layer_excite:
                dim_skip_in, _ = dim_pairs[ind + num_skip_layers_excite]
                skip_squeeze_excite = SqueezeExcite(dim_in, dim_skip_in)

            # multi-scale rgb input to feature dimension

            from_rgb = nn.Conv2d(channels, dim_in, 7, padding = 3)

            # residual convolution

            residual_conv = nn.Conv2d(dim_in, dim_out, 1, stride = (2 if should_downsample else 1))

            # main resnet block

            resnet_block = nn.Sequential(
                conv2d_3x3(dim_in, dim_out),
                leaky_relu(),
                conv2d_3x3(dim_out, dim_out),
                leaky_relu()
            )

            # multi-scale output

            multiscale_output_predictor = None

            if has_multiscale_output:
                multiscale_output_predictor = Predictor(dim_out, num_conv_kernels = num_conv_kernels, depth = 2, unconditional = unconditional)
                predictor_dims.extend([dim_out, dim_kernel_attn])

            aux_recon_decoder = None

            if has_aux_recon_decoder:
                patch_dim, frac_patches = self.aux_recon_resolutions_to_patches[resolution]

                aux_recon_decoder = SimpleDecoder(
                    dim_out,
                    dims = tuple(upsample_dims),
                    patch_dim = patch_dim,
                    frac_patches = frac_patches,
                    dropout = aux_recon_fmap_dropout
                )

            self.layers.append(nn.ModuleList([
                skip_squeeze_excite,
                from_rgb,
                resnet_block,
                residual_conv,
                SelfAttentionBlock(dim_out, heads = attn_heads, dim_head = attn_dim_head, ff_mult = ff_mult, dot_product = self_attn_dot_product) if has_attn else None,
                multiscale_output_predictor,
                aux_recon_decoder,
                Downsample(dim_out) if should_downsample else None,
            ]))

        self.to_logits = nn.Sequential(
            conv2d_3x3(dim_last, dim_last),
            leaky_relu(),
            Rearrange('b c h w -> b (c h w)'),
            nn.Linear(dim_last * (4 ** 2), 1),
            Rearrange('b 1 -> b')
        )

        # take care of text conditioning in the multiscale predictor branches

        assert unconditional or (exists(text_dim) ^ exists(text_encoder))

        if not unconditional:
            if isinstance(text_encoder, dict):
                text_encoder = TextEncoder(**text_encoder)

            self.text_dim = default(text_dim, text_encoder.dim)

            self.predictor_dims = predictor_dims
            self.text_to_conv_conditioning = nn.Linear(self.text_dim, sum(predictor_dims)) if exists(self.text_dim) else None

        self.text_encoder = text_encoder

        self.apply(self.init_)

    def init_(self, m):
        if type(m) in {nn.Conv2d, nn.Linear}:
            nn.init.kaiming_normal_(m.weight, a = 0, mode = 'fan_in', nonlinearity = 'leaky_relu')

    def resize_image_to(self, images, resolution):
        return F.interpolate(images, resolution, mode = self.resize_mode)

    @property
    def total_params(self):
        return sum([p.numel() for p in self.parameters()])

    @property
    def device(self):
        return next(self.parameters()).device

    @beartype
    def forward(
        self,
        images,
        rgbs: Optional[List[Tensor]] = None,  # multi-resolution inputs (rgbs) from the generator
        texts: Optional[List[str]] = None,
        text_encodings: Optional[Tensor] = None,
        text_embeds = None,
        real_images = None,                   # if this were passed in, the network will automatically append the real to the presumably generated images passed in as the first argument, and generate all intermediate resolutions through resizing and concat appropriately
        return_multiscale_outputs = True,     # can force it not to return multi-scale logits
        calc_aux_loss = True
    ):
        if not self.unconditional:
            assert (exists(texts) ^ exists(text_encodings)) ^ exists(text_embeds), 'either texts as List[str] is passed in, or clip text_encodings as Tensor'

            if exists(texts):
                assert exists(self.text_encoder)
                text_embeds, *_ = self.text_encoder(texts = texts)

            elif exists(text_encodings):
                assert exists(self.text_encoder)
                text_embeds, *_ = self.text_encoder(text_encodings = text_encodings)

            assert exists(text_embeds), 'raw text or text embeddings were not passed into discriminator for conditional training'

            conv_mods = self.text_to_conv_conditioning(text_embeds).split(self.predictor_dims, dim = -1)
            conv_mods = iter(conv_mods)

        else:
            assert not any([*map(exists, (texts, text_embeds))])

        x = images

        image_size = (self.image_size, self.image_size)
        assert x.shape[-2:] == image_size

        batch = x.shape[0]

        # index the rgbs by resolution

        rgbs_index = {t.shape[-1]: t for t in rgbs} if exists(rgbs) else {}

        # hold multiscale outputs

        multiscale_outputs = []

        # hold auxiliary recon losses

        aux_recon_losses = []

        # excitations

        excitations = [None] * (self.num_skip_layers_excite + 1) # +1 since first image in pixel space is not excited

        for squeeze_excite, from_rgb, block, residual_fn, attn, predictor, recon_decoder, downsample in self.layers:
            resolution = x.shape[-1]

            if exists(squeeze_excite):
                skip_excite = squeeze_excite(x)
                excitations.append(skip_excite)

            excite = safe_unshift(excitations)

            if exists(excite):
                excite = repeat(excite, 'b ... -> (s b) ...', s = x.shape[0] // excite.shape[0])
                x = x * excite

            batch_prev_stage = x.shape[0]
            has_multiscale_input = resolution in self.multiscale_input_resolutions

            if has_multiscale_input:
                rgb = rgbs_index.get(resolution, None)

                # if no rgbs passed in, assume all real images, and just resize, though realistically you would concat fake and real images together using helper function `create_real_fake_rgbs` function

                if not exists(rgb):
                    rgb = self.resize_image_to(images, resolution)

                # multi-scale input features

                multi_scale_input_feats = from_rgb(rgb)

                # expand multi-scale input features, as could include extra scales from previous stage

                multi_scale_input_feats = repeat(multi_scale_input_feats, 'b ... -> (s b) ...', s = x.shape[0] // rgb.shape[0])

                # add the multi-scale input features to the current hidden state from main stem

                x = x + multi_scale_input_feats

                # and also concat for scale invariance

                x = torch.cat((x, multi_scale_input_feats), dim = 0)

            residual = residual_fn(x)
            x = block(x)

            if exists(attn):
                x = attn(x)

            if exists(predictor):
                pred_kwargs = dict()
                if not self.unconditional:
                    pred_kwargs = dict(mod = next(conv_mods), kernel_mod = next(conv_mods))

                if return_multiscale_outputs:
                    predictor_input = x[:batch_prev_stage]
                    multiscale_outputs.append(predictor(predictor_input, **pred_kwargs))

            if exists(downsample):
                x = downsample(x)

            x = x + residual
            x = x * self.residual_scale

            if exists(recon_decoder) and calc_aux_loss:

                recon_output = x[:batch_prev_stage]
                recon_output = rearrange(x, '(s b) ... -> s b ...', b = batch)

                aux_recon_target = images

                # only use the input real images for aux recon

                recon_output = recon_output[0]

                # only reconstruct a fraction of images across batch and scale
                # for efficiency

                aux_recon_loss = recon_decoder(recon_output, aux_recon_target)
                aux_recon_losses.append(aux_recon_loss)

        # sanity check

        assert self.unconditional or is_empty([*conv_mods]), 'convolutions were incorrectly modulated'

        # to logits

        logits = self.to_logits(x)   
        logits = rearrange(logits, '(s b) ... -> s b ...', b = batch)

        return logits, multiscale_outputs, aux_recon_losses

# gan

TrainDiscrLosses = namedtuple('TrainDiscrLosses', [
    'divergence',
    'multiscale_divergence',
    'vision_aided_divergence',
    'total_matching_aware_loss',
    'gradient_penalty',
    'aux_reconstruction'
])

TrainGenLosses = namedtuple('TrainGenLosses', [
    'divergence',
    'multiscale_divergence',
    'total_vd_divergence',
    'contrastive_loss'
])

class GigaGAN(nn.Module):
    @beartype
    def __init__(
        self,
        *,
        generator: Union[BaseGenerator, Dict],
        discriminator: Union[Discriminator, Dict],
        vision_aided_discriminator: Optional[Union[VisionAidedDiscriminator, Dict]] = None,
        learning_rate = 2e-4,
        betas = (0.5, 0.9),
        weight_decay = 0.,
        discr_aux_recon_loss_weight = 1.,
        multiscale_divergence_loss_weight = 0.1,
        vision_aided_divergence_loss_weight = 0.5,
        generator_contrastive_loss_weight = 0.1,
        matching_awareness_loss_weight = 0.1,
        calc_multiscale_loss_every = 1,
        apply_gradient_penalty_every = 4,
        ttur_mult = 1,
        train_upsampler = False,
        upsampler_replace_rgb_with_input_lowres_image = False,
        log_steps_every = 20,
        create_ema_generator_at_init = True,
        save_and_sample_every = 1000,
        early_save_thres_steps = 2500,
        early_save_and_sample_every = 100,
        num_samples = 25,
        model_folder = './gigagan-models',
        results_folder = './gigagan-results',
        sample_upsampler_dl: Optional[DataLoader] = None,
        accelerator: Optional[Accelerator] = None,
        accelerate_kwargs: dict = {},
        find_unused_parameters = True,
        amp = False,
        mixed_precision_type = 'fp16'
    ):
        super().__init__()

        # create accelerator

        if accelerator:
            self.accelerator = accelerator
            assert is_empty(accelerate_kwargs)
        else:
            kwargs = DistributedDataParallelKwargs(find_unused_parameters = find_unused_parameters)

            self.accelerator = Accelerator(
                kwargs_handlers = [kwargs],
                mixed_precision = mixed_precision_type if amp else 'no',
                **accelerate_kwargs
            )

        # whether to train upsampler or not

        self.train_upsampler = train_upsampler

        if train_upsampler:
            from gigagan_pytorch.unet_upsampler import UnetUpsampler
            generator_klass = UnetUpsampler
        else:
            generator_klass = Generator

        self.upsampler_replace_rgb_with_input_lowres_image = upsampler_replace_rgb_with_input_lowres_image

        # gradient penalty and auxiliary recon loss

        self.apply_gradient_penalty_every = apply_gradient_penalty_every
        self.calc_multiscale_loss_every = calc_multiscale_loss_every

        if isinstance(generator, dict):
            generator = generator_klass(**generator)

        if isinstance(discriminator, dict):
            discriminator = Discriminator(**discriminator)

        if exists(vision_aided_discriminator) and isinstance(vision_aided_discriminator, dict):
            vision_aided_discriminator = VisionAidedDiscriminator(**vision_aided_discriminator)

        assert isinstance(generator, generator_klass)

        # use _base to designate unwrapped models

        self.G = generator
        self.D = discriminator
        self.VD = vision_aided_discriminator

        # ema

        self.has_ema_generator = False

        if self.is_main and create_ema_generator_at_init:
            self.create_ema_generator()

        # print number of parameters

        self.print(f'Generator: {numerize.numerize(generator.total_params)}')
        self.print(f'Discriminator: {numerize.numerize(discriminator.total_params)}')

        if exists(self.VD):
            self.print(f'Vision Discriminator: {numerize.numerize(vision_aided_discriminator.total_params)}')

        self.print('\n')

        # text encoder

        assert generator.unconditional == discriminator.unconditional
        assert not exists(vision_aided_discriminator) or vision_aided_discriminator.unconditional == generator.unconditional

        self.unconditional = generator.unconditional

        # optimizers

        self.G_opt = get_optimizer(self.G.parameters(), lr = learning_rate, betas = betas, weight_decay = weight_decay)
        self.D_opt = get_optimizer(self.D.parameters(), lr = learning_rate * ttur_mult, betas = betas, weight_decay = weight_decay)

        # prepare for distributed

        self.G, self.D, self.G_opt, self.D_opt = self.accelerator.prepare(self.G, self.D, self.G_opt, self.D_opt)

        # vision aided discriminator optimizer

        if exists(self.VD):
            self.VD_opt = get_optimizer(self.VD.parameters(), lr = learning_rate, betas = betas, weight_decay = weight_decay)
            self.VD_opt = self.accelerator.prepare(self.VD_opt)

        # loss related

        self.discr_aux_recon_loss_weight = discr_aux_recon_loss_weight
        self.multiscale_divergence_loss_weight = multiscale_divergence_loss_weight
        self.vision_aided_divergence_loss_weight = vision_aided_divergence_loss_weight
        self.generator_contrastive_loss_weight = generator_contrastive_loss_weight
        self.matching_awareness_loss_weight = matching_awareness_loss_weight

        # steps

        self.log_steps_every = log_steps_every

        self.register_buffer('steps', torch.ones(1, dtype = torch.long))

        # save and sample

        self.save_and_sample_every = save_and_sample_every
        self.early_save_thres_steps = early_save_thres_steps
        self.early_save_and_sample_every = early_save_and_sample_every

        self.num_samples = num_samples

        self.train_dl = None

        self.sample_upsampler_dl_iter = None
        if exists(sample_upsampler_dl):
            self.sample_upsampler_dl_iter = cycle(self.sample_upsampler_dl)

        self.results_folder = Path(results_folder)
        self.model_folder = Path(model_folder)

        mkdir_if_not_exists(self.results_folder)
        mkdir_if_not_exists(self.model_folder)

    def save(self, path, overwrite = True):
        path = Path(path)
        mkdir_if_not_exists(path.parents[0])

        assert overwrite or not path.exists()

        pkg = dict(
            G = self.unwrapped_G.state_dict(),
            D = self.unwrapped_D.state_dict(),
            G_opt = self.G_opt.state_dict(),
            D_opt = self.D_opt.state_dict(),
            steps = self.steps.item(),
            version = __version__
        )

        if exists(self.G_opt.scaler):
            pkg['G_scaler'] = self.G_opt.scaler.state_dict()

        if exists(self.D_opt.scaler):
            pkg['D_scaler'] = self.D_opt.scaler.state_dict()

        if exists(self.VD):
            pkg['VD'] = self.unwrapped_VD.state_dict()
            pkg['VD_opt'] = self.VD_opt.state_dict()

            if exists(self.VD_opt.scaler):
                pkg['VD_scaler'] = self.VD_opt.scaler.state_dict()

        if self.has_ema_generator:
            pkg['G_ema'] = self.G_ema.state_dict()

        torch.save(pkg, str(path))

    def load(self, path, strict = False):
        path = Path(path)
        assert path.exists()

        pkg = torch.load(str(path))

        if 'version' in pkg and pkg['version'] != __version__:
            print(f"trying to load from version {pkg['version']}")

        self.unwrapped_G.load_state_dict(pkg['G'], strict = strict)
        self.unwrapped_D.load_state_dict(pkg['D'], strict = strict)

        if exists(self.VD):
            self.unwrapped_VD.load_state_dict(pkg['VD'], strict = strict)

        if self.has_ema_generator:
            self.G_ema.load_state_dict(pkg['G_ema'])

        if 'steps' in pkg:
            self.steps.copy_(torch.tensor([pkg['steps']]))

        if 'G_opt'not in pkg or 'D_opt' not in pkg:
            return

        try:
            self.G_opt.load_state_dict(pkg['G_opt'])
            self.D_opt.load_state_dict(pkg['D_opt'])

            if exists(self.VD):
                self.VD_opt.load_state_dict(pkg['VD_opt'])

            if 'G_scaler' in pkg and exists(self.G_opt.scaler):
                self.G_opt.scaler.load_state_dict(pkg['G_scaler'])

            if 'D_scaler' in pkg and exists(self.D_opt.scaler):
                self.D_opt.scaler.load_state_dict(pkg['D_scaler'])

            if 'VD_scaler' in pkg and exists(self.VD_opt.scaler):
                self.VD_opt.scaler.load_state_dict(pkg['VD_scaler'])

        except Exception as e:
            self.print(f'unable to load optimizers {e.msg}- optimizer states will be reset')
            pass

    # accelerate related

    @property
    def device(self):
        return self.accelerator.device

    @property
    def unwrapped_G(self):
        return self.accelerator.unwrap_model(self.G)

    @property
    def unwrapped_D(self):
        return self.accelerator.unwrap_model(self.D)

    @property
    def unwrapped_VD(self):
        return self.accelerator.unwrap_model(self.VD)

    def print(self, msg):
        self.accelerator.print(msg)

    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    @beartype
    def set_dataloader(self, dl: DataLoader):
        assert not exists(self.train_dl), 'training dataloader has already been set'

        self.train_dl = dl
        self.train_dl_batch_size = dl.batch_size

        self.train_dl = self.accelerator.prepare(self.train_dl)

    # create EMA generator

    def create_ema_generator(
        self,
        update_every = 10,
        update_after_step = 100,
        decay = 0.995
    ):
        if not self.is_main:
            return

        assert not self.has_ema_generator, 'EMA generator has already been created'

        self.G_ema = EMA(self.unwrapped_G, update_every = update_every, update_after_step = update_after_step, beta = decay)
        self.has_ema_generator = True

    def generate_kwargs(self, dl_iter, batch_size):
        # what to pass into the generator
        # depends on whether training upsampler or not

        maybe_text_kwargs = dict()
        if self.train_upsampler or not self.unconditional:
            assert exists(dl_iter)

            if self.unconditional:
                real_images = next(dl_iter)
            else:
                result = next(dl_iter)
                assert isinstance(result, tuple), 'dataset should return a tuple of two items for text conditioned training, (images: Tensor, texts: List[str])'
                real_images, texts = result

                maybe_text_kwargs['texts'] = texts[:batch_size]

            real_images = real_images.to(self.device)

        # if training upsample generator, need to downsample real images

        if self.train_upsampler:
            size = self.G.input_image_size
            lowres_real_images = F.interpolate(real_images, (size, size))

            G_kwargs = dict(
                lowres_image = lowres_real_images,
                replace_rgb_with_input_lowres_image = self.upsampler_replace_rgb_with_input_lowres_image
            )
        else:
            assert exists(batch_size)

            G_kwargs = dict(batch_size = batch_size)

        # create noise

        noise = torch.randn(batch_size, self.unwrapped_G.style_network.dim, device = self.device)

        G_kwargs.update(noise = noise)

        return G_kwargs, maybe_text_kwargs
    
    @beartype
    def train_discriminator_step(
        self,
        dl_iter: Iterable,
        grad_accum_every = 1,
        apply_gradient_penalty = False,
        calc_multiscale_loss = True
    ):
        total_divergence = 0.
        total_vision_aided_divergence = 0.

        total_gp_loss = 0.
        total_aux_loss = 0.

        total_multiscale_divergence = 0. if calc_multiscale_loss else None

        has_matching_awareness = not self.unconditional and self.matching_awareness_loss_weight > 0.

        total_matching_aware_loss = 0.

        all_texts = []
        all_fake_images = []
        all_fake_rgbs = []
        all_real_images = []

        self.G.train()
        self.D.train()

        self.D_opt.zero_grad()

        for _ in range(grad_accum_every):

            if self.unconditional:
                real_images = next(dl_iter)
            else:
                result = next(dl_iter)
                assert isinstance(result, tuple), 'dataset should return a tuple of two items for text conditioned training, (images: Tensor, texts: List[str])'
                real_images, texts = result

                all_real_images.append(real_iamges)
                all_texts.extend(texts)

            # requires grad for real images, for gradient penalty

            real_images = real_images.to(self.device)
            real_images.requires_grad_()

            batch_size = real_images.shape[0]

            # for discriminator training, fit upsampler and image synthesis logic under same function

            G_kwargs, maybe_text_kwargs = self.generate_kwargs(dl_iter, batch_size)

            # generator

            with torch.no_grad(), self.accelerator.autocast():
                images, rgbs = self.G(
                    **G_kwargs,
                    **maybe_text_kwargs,
                    return_all_rgbs = True
                )

                all_fake_images.append(images)
                all_fake_rgbs = append(rgbs)

                # detach output of generator, as training discriminator only

                images.detach_()

                for rgb in rgbs:
                    rgb.detach_()

            # main divergence loss

            with self.accelerator.autocast():
                fake_logits, fake_multiscale_logits, _ = self.D(
                    images,
                    rgbs,
                    **maybe_text_kwargs,
                    return_multiscale_outputs = calc_multiscale_loss,
                    calc_aux_loss = False
                )

                real_logits, real_multiscale_logits, aux_recon_losses = self.D(
                    real_images,
                    **maybe_text_kwargs,
                    return_multiscale_outputs = calc_multiscale_loss,
                    calc_aux_loss = True
                )

                divergence = discriminator_hinge_loss(real_logits, fake_logits)
                total_divergence += (divergence.item() / grad_accum_every)

                # handle multi-scale divergence

                multiscale_divergence = 0.

                if self.multiscale_divergence_loss_weight > 0. and len(fake_multiscale_logits) > 0:

                    for multiscale_fake, multiscale_real in zip(fake_multiscale_logits, real_multiscale_logits):
                        multiscale_loss = discriminator_hinge_loss(multiscale_real, multiscale_fake)
                        multiscale_divergence = multiscale_divergence + multiscale_loss

                    total_multiscale_divergence += (multiscale_divergence.item() / grad_accum_every)

                # figure out gradient penalty if needed

                gp_loss = 0.

                if apply_gradient_penalty:
                    gp_loss = gradient_penalty(
                        real_images,
                        outputs = [real_logits, *real_multiscale_logits],
                        grad_output_weights = [1., *(self.multiscale_divergence_loss_weight,) * len(real_multiscale_logits)],
                        scaler = self.D_opt.scaler
                    )

                    total_gp_loss += (gp_loss.item() / grad_accum_every)

                # handle vision aided discriminator, if needed

                vd_loss = 0.

                if exists(self.VD):
                    fake_vision_aided_logits = self.VD(images, **maybe_text_kwargs)
                    real_vision_aided_logits = self.VD(real_images, **maybe_text_kwargs)

                    for fake_logits, real_logits in zip(fake_vision_aided_logits, real_vision_aided_logits):
                        vd_loss = vd_loss + discriminator_hinge_loss(real_logits, fake_logits)

                    total_vision_aided_divergence += (vd_loss.item() / grad_accum_every)

                    # handle gradient penalty for vision aided discriminator

                    if apply_gradient_penalty:
                        vd_gp_loss = gradient_penalty(
                            real_images,
                            outputs = real_vision_aided_logits,
                            grad_output_weights = [self.vision_aided_divergence_loss_weight] * len(real_vision_aided_logits),
                            scaler = self.VD_opt.scaler
                        )

                        gp_loss = gp_loss + vd_gp_loss

                        total_gp_loss += (vd_gp_loss.item() / grad_accum_every)

                # sum up losses

                total_loss = divergence + gp_loss

                if self.multiscale_divergence_loss_weight > 0.:
                    total_loss = total_loss + multiscale_divergence * self.multiscale_divergence_loss_weight

                if self.vision_aided_divergence_loss_weight > 0.:
                    total_loss = total_loss + vd_loss * self.vision_aided_divergence_loss_weight

                if self.discr_aux_recon_loss_weight > 0.:
                    aux_loss = sum(aux_recon_losses)

                    total_aux_loss += (aux_loss.item() / grad_accum_every)

                    total_loss = total_loss + aux_loss * self.discr_aux_recon_loss_weight

            # backwards

            self.accelerator.backward(total_loss / grad_accum_every)


        # matching awareness loss
        # strategy would be to rotate the texts by one and assume batch is shuffled enough for mismatched conditions

        if has_matching_awareness:
            all_real_images = torch.cat(all_real_images, dim = 0)
            all_fake_images = torch.cat(all_fake_images, dim = 0)
            all_fake_rgbs = torch.cat(all_fake_rgbs, dim = 0)

            # rotate texts

            all_texts = [*all_texts[1:], all_texts[0]]

            zipped_data = zip(
                all_fake_images.split(batch_size, dim = 0),
                all_fake_rgbs.split(batch_size, dim = 0),
                all_real_images.split(batch_size, dim = 0),
                group_by_num_consecutive(texts, batch_size)
            )

            total_loss = 0.

            for fake_images, fake_rgbs, real_images, texts in zipped_data:
                with torch.accelerator.autocast():
                    fake_logits, *_ = self.D(
                        fake_images,
                        fake_rgbs,
                        texts = texts,
                        calc_multiscale_loss = False,
                        calc_aux_loss = False
                    )

                    real_logits, *_ = self.D(
                        real_images,
                        texts = texts,
                        calc_multiscale_loss = False,
                        calc_aux_loss = False
                    )

                    matching_loss = aux_matching_loss(real_logits, fake_logits)

                    total_matching_aware_loss = (matching_loss.item() / grad_accum_every)

                    loss = matching_loss * self.matching_awareness_loss_weight

                self.accelerator.backward(loss / grad_accum_every)

        self.D_opt.step()

        return TrainDiscrLosses(
            total_divergence,
            total_multiscale_divergence,
            total_vision_aided_divergence,
            total_matching_aware_loss,
            total_gp_loss,
            total_aux_loss
        )

    def train_generator_step(
        self,
        batch_size = None,
        dl_iter: Optional[Iterable] = None,
        grad_accum_every = 1,
        calc_multiscale_loss = True
    ):
        need_contrastive_loss = self.generator_contrastive_loss_weight > 0. and not self.unconditional

        total_divergence = 0.
        total_multiscale_divergence = 0. if calc_multiscale_loss else None
        total_vd_divergence = 0.
        contrastive_loss = 0.

        self.G.train()
        self.D.train()

        self.D_opt.zero_grad()
        self.G_opt.zero_grad()

        all_images = []
        all_texts = []

        for _ in range(grad_accum_every):

            # generator
            
            G_kwargs, maybe_text_kwargs = self.generate_kwargs(dl_iter, batch_size)

            with self.accelerator.autocast():
                image, rgbs = self.G(
                    **G_kwargs,
                    **maybe_text_kwargs,
                    return_all_rgbs = True
                )

                # accumulate all images and texts for maybe contrastive loss

                if need_contrastive_loss:
                    all_images.append(image)
                    all_texts.extend(maybe_text_kwargs['texts'])

                # discriminator

                logits, multiscale_logits, _ = self.D(
                    image,
                    rgbs,
                    **maybe_text_kwargs,
                    return_multiscale_outputs = calc_multiscale_loss,
                    calc_aux_loss = False
                )

                # generator hinge loss discriminator and multiscale

                divergence = generator_hinge_loss(logits)

                total_divergence += (divergence.item() / grad_accum_every)

                total_loss = divergence

                if self.multiscale_divergence_loss_weight > 0. and len(multiscale_logits) > 0:
                    multiscale_divergence = 0.

                    for multiscale_logit in multiscale_logits:
                        multiscale_divergence = multiscale_divergence + generator_hinge_loss(multiscale_logit)

                    total_multiscale_divergence += (multiscale_divergence.item() / grad_accum_every)

                    total_loss = total_loss + multiscale_divergence * self.multiscale_divergence_loss_weight

                # vision aided generator hinge loss

                if exists(self.VD) and self.vision_aided_divergence_loss_weight > 0.:
                    vd_loss = 0.

                    logits = self.VD(image, **maybe_text_kwargs)

                    for logit in logits:
                        vd_loss = vd_loss + generator_hinge_loss(logits)

                    total_vd_divergence += (vd_loss.item() / grad_accum_every)

                    total_loss = total_loss + vd_loss * self.vision_aided_divergence_loss_weight

            self.accelerator.backward(total_loss / grad_accum_every, retain_graph = need_contrastive_loss)

        # if needs the generator contrastive loss
        # gather up all images and texts and calculate it

        if need_contrastive_loss:
            all_images = torch.cat(all_images, dim = 0)

            contrastive_loss = aux_clip_loss(
                clip = self.G.text_encoder.clip,
                texts = all_texts,
                images = all_images
            )

            self.accelerator.backward(contrastive_loss * self.generator_contrastive_loss_weight)

        # generator optimizer step

        self.G_opt.step()

        # update exponentially moving averaged generator

        self.accelerator.wait_for_everyone()

        if self.is_main and self.has_ema_generator:
            self.G_ema.update()

        return TrainGenLosses(
            total_divergence,
            total_multiscale_divergence,
            total_vd_divergence,
            contrastive_loss
        )

    def sample(self, dl_iter, batch_size):
        G_kwargs, maybe_text_kwargs = self.generate_kwargs(dl_iter, batch_size)

        with self.accelerator.autocast():
            generator_output = self.G(**G_kwargs, **maybe_text_kwargs)

        if not self.train_upsampler:
            return generator_output

        output_size = generator_output.shape[-1]
        lowres_image = G_kwargs['lowres_image']
        lowres_image = F.interpolate(lowres_image, (output_size, output_size))

        return torch.cat([lowres_image, generator_output])

    @torch.inference_mode()
    def save_sample(
        self,
        batch_size,
        dl_iter = None
    ):
        milestone = self.steps.item() // self.save_and_sample_every
        nrow_mult = 2 if self.train_upsampler else 1
        batches = num_to_groups(self.num_samples, batch_size)

        if self.train_upsampler:
            dl_iter = default(self.sample_upsampler_dl_iter, dl_iter)

        assert exists(dl_iter)

        sample_models_and_output_file_name = [(self.G, f'sample-{milestone}.png')]

        if self.has_ema_generator:
            sample_models_and_output_file_name.append((self.G_ema, f'ema-sample-{milestone}.png'))

        for model, filename in sample_models_and_output_file_name:
            model.eval()

            all_images_list = list(map(lambda n: self.sample(dl_iter, n), batches))
            all_images = torch.cat(all_images_list, dim=0)

            all_images.clamp_(0., 1.)

            utils.save_image(
                all_images,
                str(self.results_folder / filename),
                nrow = int(sqrt(self.num_samples)) * nrow_mult
            )

        # Possible to do: Include some metric to save if improved, include some sampler dict text entries
        self.save(str(self.model_folder / f'model-{milestone}.ckpt'))

    @beartype
    def forward(
        self,
        *,
        steps,
        grad_accum_every = 1
    ):
        assert exists(self.train_dl), 'you need to set the dataloader by running .set_dataloader(dl: Dataloader)'

        batch_size = self.train_dl_batch_size
        dl_iter = cycle(self.train_dl)

        last_gp_loss = 0.
        last_multiscale_d_loss = 0.
        last_multiscale_g_loss = 0.

        for _ in tqdm(range(steps), initial = self.steps.item()):
            steps = self.steps.item()
            is_first_step = steps == 1

            apply_gradient_penalty = self.apply_gradient_penalty_every > 0 and divisible_by(steps, self.apply_gradient_penalty_every)
            calc_multiscale_loss =  self.calc_multiscale_loss_every > 0 and divisible_by(steps, self.calc_multiscale_loss_every)

            (
                d_loss,
                multiscale_d_loss,
                vision_aided_d_loss,
                gp_loss,
                recon_loss
            ) = self.train_discriminator_step(
                dl_iter = dl_iter,
                grad_accum_every = grad_accum_every,
                apply_gradient_penalty = apply_gradient_penalty,
                calc_multiscale_loss = calc_multiscale_loss
            )

            self.accelerator.wait_for_everyone()

            (
                g_loss,
                multiscale_g_loss,
                vision_aided_g_loss,
                matching_aware_loss,
                contrastive_loss
            ) = self.train_generator_step(
                dl_iter = dl_iter,
                batch_size = batch_size,
                grad_accum_every = grad_accum_every,
                calc_multiscale_loss = calc_multiscale_loss
            )

            if exists(gp_loss):
                last_gp_loss = gp_loss

            if exists(multiscale_d_loss):
                last_multiscale_d_loss = multiscale_d_loss

            if exists(multiscale_g_loss):
                last_multiscale_g_loss = multiscale_g_loss

            if is_first_step or divisible_by(steps, self.log_steps_every):
                self.print(f' G: {g_loss:.2f} | MSG: {last_multiscale_g_loss:.2f} | VG: {vision_aided_g_loss:.2f} | D: {d_loss:.2f} | MSD: {last_multiscale_d_loss:.2f} | VD: {vision_aided_d_loss:.2f} | GP: {last_gp_loss:.2f} | SSL: {recon_loss:.2f} | CL: {contrastive_loss:.2f} | MAL: {matching_aware_loss:.2f}')

            self.accelerator.wait_for_everyone()

            if self.is_main and (is_first_step or divisible_by(steps, self.save_and_sample_every) or (steps <= self.early_save_thres_steps and divisible_by(steps, self.early_save_and_sample_every))):
                self.save_sample(batch_size, dl_iter)
            
            self.steps += 1

        self.print(f'complete {steps} training steps')

import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, pack, unpack, repeat

# helpers

def exists(val):
    return val is not None

# attention
# they use an attention with a better Lipchitz constant - l2 distance similarity instead of dot product - shown in vitgan to be more stable

class Attention(nn.Module):
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

        self.to_qkv = nn.Conv2d(dim, dim_inner * 3, 1, bias = False)
        self.to_out = nn.Conv2d(dim_inner, dim, 1, bias = False)

    def forward(self, fmap):
        x, y = fmap.shape[-2:]

        h = self.heads

        q, k, v = self.to_qkv(fmap).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> (b h) (x y) d', h = self.heads), (q, k, v))

        sim = torch.cdist(q, k, p = 2) # l2 distance

        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)

        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', x = x, y = y, h = h)

        return self.to_out(out)

# gan

class GigaGAN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

import torch
import torch.nn.functional as F
from torch.autograd import Function
import torch.distributed as dist

from einops import rearrange

# helpers

def exists(val):
    return val is not None

def pad_dim_to(t, length, dim = 0):
    pad_length = length - t.shape[dim]
    zero_pairs = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    return F.pad(t, (*((0, 0) * zero_pairs), 0, pad_length))

# distributed helpers

def all_gather_variable_dim(t, dim = 0, sizes = None):
    device, world_size = t.device, dist.get_world_size()

    if not exists(sizes):
        size = torch.tensor(t.shape[dim], device = device, dtype = torch.long)
        sizes = [torch.empty_like(size, device = device, dtype = torch.long) for i in range(world_size)]
        dist.all_gather(sizes, size)
        sizes = torch.stack(sizes)

    max_size = sizes.amax().item()
    padded_t = pad_dim_to(t, max_size, dim = dim)

    gathered_tensors = [torch.empty(padded_t.shape, device = device, dtype = padded_t.dtype) for i in range(world_size)]
    dist.all_gather(gathered_tensors, padded_t)

    gathered_tensor = torch.cat(gathered_tensors, dim = dim)
    seq = torch.arange(max_size, device = device)

    mask = rearrange(seq, 'j -> 1 j') < rearrange(sizes, 'i -> i 1')
    mask = rearrange(mask, 'i j -> (i j)')
    seq = torch.arange(mask.shape[-1], device = device)
    indices = seq[mask]

    gathered_tensor = gathered_tensor.index_select(dim, indices)

    return gathered_tensor, sizes

class AllGather(Function):
    @staticmethod
    def forward(ctx, x, dim, sizes):
        is_dist = dist.is_initialized() and dist.get_world_size() > 1
        ctx.is_dist = is_dist

        if not is_dist:
            return x, None

        x, batch_sizes = all_gather_variable_dim(x, dim = dim, sizes = sizes)
        ctx.batch_sizes = batch_sizes.tolist()
        ctx.dim = dim
        return x, batch_sizes

    @staticmethod
    def backward(ctx, grads, _):
        if not ctx.is_dist:
            return grads, None, None

        batch_sizes, rank = ctx.batch_sizes, dist.get_rank()
        grads_by_rank = grads.split(batch_sizes, dim = ctx.dim)
        return grads_by_rank[rank], None, None

all_gather = AllGather.apply

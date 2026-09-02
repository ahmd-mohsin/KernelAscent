import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 638
M, D, DT = 1024, 4096, torch.bfloat16


@triton.jit
def _fused_softmax_ln_relu(
    x_ptr, g_ptr, b_ptr, out_ptr,
    N, eps,
    S1, S2,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    base = row * N

    # load input (bf16 -> fp32)
    x = tl.load(x_ptr + base + offs, mask=mask, other=-float('inf')).to(tl.float32)

    # x = x * 1.1576  (bf16 rounding as in reference)
    x = (x * S1).to(tl.bfloat16).to(tl.float32)

    # softmax (fp32 accumulation, bf16 output as in reference)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = (e / s).to(tl.bfloat16).to(tl.float32)

    # y = y * 1.3589 (bf16 rounding)
    y = (y * S2).to(tl.bfloat16).to(tl.float32)

    # layernorm (fp32 statistics)
    mean = tl.sum(tl.where(mask, y, 0.0), axis=0) / N
    d = tl.where(mask, y - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    inv = 1.0 / tl.sqrt(var + eps)

    g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    z = d * inv * g + b

    # relu
    z = tl.maximum(z, 0.0)

    tl.store(out_ptr + base + offs, z.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln3_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x * 1.1576
            x = torch.softmax(x, dim=-1)
            x = x * 1.3589
            x = F.layer_norm(x, (x.shape[-1],), self.ln3_g, self.ln3_b)
            return torch.relu(x)

        x = x.contiguous()
        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.view(-1, N)
        rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        _fused_softmax_ln_relu[(rows,)](
            x2, self.ln3_g, self.ln3_b, out,
            N, 1e-5,
            1.1576, 1.3589,
            BLOCK=BLOCK,
            num_warps=16 if BLOCK >= 4096 else 8,
        )
        return out.view(orig_shape)

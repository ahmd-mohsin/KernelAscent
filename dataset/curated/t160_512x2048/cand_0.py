import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 160
M, D, DT = 512, 2048, torch.bfloat16


@triton.jit
def _fused_ln_softmax_kernel(
    X, G, B, B1, B2, OUT,
    stride_x, stride_o,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm (fp32 math, matching PyTorch's bf16 layer_norm internals)
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    invstd = 1.0 / tl.sqrt(var + 1e-5)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x - mean) * invstd * g + b
    # round to bf16 as PyTorch does at op boundaries
    y = y.to(tl.bfloat16)

    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.bfloat16)
    y = (y + b1).to(tl.bfloat16)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.bfloat16)
    y = (y + b2).to(tl.bfloat16)

    # scalar multiply: bf16 tensor * float scalar -> fp32 opmath -> bf16
    z = (y.to(tl.float32) * 1.0242).to(tl.bfloat16)

    # softmax with fp32 accumulation (matching PyTorch bf16 softmax)
    zf = z.to(tl.float32)
    zf = tl.where(mask, zf, float('-inf'))
    zmax = tl.max(zf, axis=0)
    e = tl.exp(zf - zmax)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.bfloat16)

    tl.store(OUT + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
            y = y + self.b1
            y = y + self.b2
            y = y * 1.0242
            return torch.softmax(y, dim=-1)

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_ln_softmax_kernel[(rows,)](
            x2, self.ln0_g, self.ln0_b, self.b1, self.b2, out,
            x2.stride(0), out.stride(0),
            N=N, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)

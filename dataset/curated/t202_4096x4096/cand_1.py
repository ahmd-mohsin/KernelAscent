import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 202
M, D, DT = 4096, 4096, torch.float16


@triton.jit
def _fused_gelu2_ln_kernel(
    X, W, B, Y,
    stride_x, stride_y,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    INV_SQRT2: tl.constexpr = 0.7071067811865476

    # gelu #1 (exact, fp32 math, round to fp16 like PyTorch does for half tensors)
    g1 = x * 0.5 * (1.0 + tl.math.erf(x * INV_SQRT2))
    g1 = g1.to(tl.float16).to(tl.float32)

    # gelu #2
    g2 = g1 * 0.5 * (1.0 + tl.math.erf(g1 * INV_SQRT2))
    g2 = g2.to(tl.float16).to(tl.float32)

    # layernorm in fp32
    g2m = tl.where(mask, g2, 0.0)
    mean = tl.sum(g2m, axis=0) / N
    diff = tl.where(mask, g2 - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = (g2 - mean) * rstd * w + b
    tl.store(Y + row * stride_y + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln2_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = F.gelu(x)
            x = F.gelu(x)
            return F.layer_norm(x, (x.shape[-1],), self.ln2_g, self.ln2_b)

        orig_shape = x.shape
        N = orig_shape[-1]
        x2d = x.contiguous().view(-1, N)
        rows = x2d.shape[0]
        y = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_gelu2_ln_kernel[(rows,)](
            x2d, self.ln2_g, self.ln2_b, y,
            x2d.stride(0), y.stride(0),
            N, 1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)

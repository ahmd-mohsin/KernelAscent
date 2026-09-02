import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 338
M, D, DT = 512, 4097, torch.bfloat16


@triton.jit
def _ln_gelu2_kernel(
    X, G, B, Y,
    N, eps,
    stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # mean / var in fp32
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)

    y = (x - mean) * rstd * g + b
    # round to bf16 (layer_norm output dtype), then gelu in fp32 like PyTorch opmath
    y = y.to(tl.bfloat16).to(tl.float32)

    SQRT1_2: tl.constexpr = 0.7071067811865476
    y = y * 0.5 * (1.0 + tl.math.erf(y * SQRT1_2))
    y = y.to(tl.bfloat16).to(tl.float32)

    y = y * 0.5 * (1.0 + tl.math.erf(y * SQRT1_2))
    y = y.to(tl.bfloat16)

    tl.store(Y + row * stride_y + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
            x = F.gelu(x)
            x = F.gelu(x)
            return x

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 4096 else 4

        _ln_gelu2_kernel[(rows,)](
            x2, self.ln0_g, self.ln0_b, y,
            N, 1e-5,
            x2.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)

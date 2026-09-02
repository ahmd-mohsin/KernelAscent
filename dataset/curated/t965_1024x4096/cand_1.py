import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 965
M, D, DT = 1024, 4096, torch.bfloat16


@triton.jit
def _fused_ln_gelu_kernel(
    X, G, B, Y,
    stride_xm, stride_ym,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    # step 1: x * 1.311, rounded to bf16 (matches PyTorch elementwise store)
    x = (x.to(tl.float32) * 1.311).to(tl.bfloat16).to(tl.float32)

    # step 2: layer norm in fp32 (PyTorch upcasts bf16 to fp32 internally)
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x - mean) * rstd * g + b
    y = y.to(tl.bfloat16).to(tl.float32)

    # step 3: exact GELU: 0.5 * y * (1 + erf(y / sqrt(2)))
    y = 0.5 * y * (1.0 + tl.math.erf(y * 0.7071067811865476))
    y = y.to(tl.bfloat16).to(tl.float32)

    # step 4: scale
    y = (y * 1.4814).to(tl.bfloat16)

    tl.store(Y + row * stride_ym + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x * 1.311
            x = F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)
            x = F.gelu(x)
            x = x * 1.4814
            return x

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 4096 else 4

        _fused_ln_gelu_kernel[(rows,)](
            x2, self.ln1_g, self.ln1_b, y,
            x2.stride(0), y.stride(0),
            N=N, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)

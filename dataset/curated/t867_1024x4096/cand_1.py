import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 867
M, D, DT = 1024, 4096, torch.bfloat16


@triton.jit
def _fused_add_ln_add_kernel(
    X, B0, G, B, B2, Y,
    N, eps,
    stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    b0 = tl.load(B0 + cols, mask=mask, other=0.0)

    # x + b0 in bf16 semantics (fp32 compute, round to bf16)
    xb = (x.to(tl.float32) + b0.to(tl.float32)).to(tl.bfloat16)
    xf = xb.to(tl.float32)

    mean = tl.sum(tl.where(mask, xf, 0.0), axis=0) / N
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (xf - mean) * rstd * g + b
    y_bf = y.to(tl.bfloat16)

    b2 = tl.load(B2 + cols, mask=mask, other=0.0)
    out = (y_bf.to(tl.float32) + b2.to(tl.float32)).to(tl.bfloat16)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x + self.b0
            x = F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)
            x = x + self.b2
            return x

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_add_ln_add_kernel[(rows,)](
            x2, self.b0, self.ln1_g, self.ln1_b, self.b2, y,
            N, 1e-5,
            x2.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)

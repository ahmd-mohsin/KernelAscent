import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 145
M, D, DT = 4096, 512, torch.bfloat16


@triton.jit
def _fused_kernel(
    X, B1, G, B, Y,
    stride_xm, stride_ym,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)  # bf16
    b1 = tl.load(B1 + cols, mask=mask, other=0.0)                  # bf16

    # relu + bias (in bf16, matching reference rounding)
    x = tl.maximum(x, 0.0)
    x = (x + b1).to(tl.bfloat16)

    # layernorm in fp32
    xf = x.to(tl.float32)
    mean = tl.sum(tl.where(mask, xf, 0.0), axis=0) / N
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (xf - mean) * rstd * g + b

    # cast to bf16 (reference materializes bf16 layernorm output), then softmax in fp32
    y = y.to(tl.bfloat16).to(tl.float32)
    y = tl.where(mask, y, float('-inf'))
    ymax = tl.max(y, axis=0)
    e = tl.exp(y - ymax)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    # relu (identity on softmax output, kept for exactness)
    out = tl.maximum(out, 0.0)

    tl.store(Y + row * stride_ym + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = torch.relu(x)
            x = x + self.b1
            x = F.layer_norm(x, (x.shape[-1],), self.ln2_g, self.ln2_b)
            x = torch.softmax(x, dim=-1)
            return torch.relu(x)

        orig_shape = x.shape
        n = orig_shape[-1]
        x2 = x.contiguous().view(-1, n)
        m = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_kernel[(m,)](
            x2, self.b1, self.ln2_g, self.ln2_b, y,
            x2.stride(0), y.stride(0),
            n, 1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)

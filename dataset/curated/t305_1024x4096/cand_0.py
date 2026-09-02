import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 305
M, D, DT = 1024, 4096, torch.bfloat16


@triton.jit
def _fused_scale_ln_relu(
    X, G, B, Y,
    N, eps, scale,
    stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    # replicate: x = (x * 1.0924) rounded to bf16, then layer_norm in fp32
    xs = (x.to(tl.float32) * scale).to(tl.bfloat16).to(tl.float32)

    mean = tl.sum(xs, axis=0) / N
    diff = tl.where(mask, xs - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = (xs - mean) * rstd * g + b
    y = tl.maximum(y, 0.0)
    tl.store(Y + row * stride_y + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x * 1.0924
            x = F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)
            return torch.relu(x)

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_scale_ln_relu[(rows,)](
            x2, self.ln1_g, self.ln1_b, y,
            N, 1e-5, 1.0924,
            x2.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)

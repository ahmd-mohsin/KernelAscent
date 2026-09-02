import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 449
M, D, DT = 1024, 1025, torch.bfloat16


@triton.jit
def _scale_ln_kernel(
    X, G, B, Y,
    N, stride_x, stride_y,
    scale, eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    # scale in fp32, round to bf16 (matches x = x * 1.2438 on bf16 tensor)
    xs = (x.to(tl.float32) * scale).to(tl.bfloat16).to(tl.float32)
    xs = tl.where(mask, xs, 0.0)

    mean = tl.sum(xs, axis=0) / N
    diff = tl.where(mask, xs - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = diff * rstd * g + b
    tl.store(Y + row * stride_y + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x * 1.2438
            return F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        Mrows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK_N = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK_N >= 2048 else 4

        _scale_ln_kernel[(Mrows,)](
            x2, self.ln1_g, self.ln1_b, y,
            N, x2.stride(0), y.stride(0),
            1.2438, 1e-5,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return y.view(orig_shape)

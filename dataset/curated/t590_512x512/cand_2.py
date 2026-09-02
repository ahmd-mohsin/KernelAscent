import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 590
M, D, DT = 512, 512, torch.bfloat16


@triton.jit
def _fused_kernel(
    X, Y, G, B, B3,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    EPS: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0)

    # x = x * 1.254 (bf16 rounding to match PyTorch)
    x = (x.to(tl.float32) * 1.254).to(tl.bfloat16)
    # relu
    x = tl.maximum(x, 0.0)

    xf = x.to(tl.float32)

    # layer norm stats in fp32
    mean = tl.sum(tl.where(mask, xf, 0.0), axis=0) / N
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)

    y = ((xf - mean) * rstd * g + b).to(tl.bfloat16)   # layer_norm output (bf16)
    y = (y.to(tl.float32) + b3).to(tl.bfloat16)        # + b3 (bf16 rounding)
    y = (y.to(tl.float32) * 1.1001).to(tl.bfloat16)    # * 1.1001 (bf16 rounding)

    tl.store(Y + row * N + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln2_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x * 1.254
            x = torch.relu(x)
            x = F.layer_norm(x, (x.shape[-1],), self.ln2_g, self.ln2_b)
            x = x + self.b3
            x = x * 1.1001
            return x

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_kernel[(rows,)](
            x2, y, self.ln2_g, self.ln2_b, self.b3,
            N=N, BLOCK=BLOCK, EPS=1e-5,
            num_warps=num_warps,
        )
        return y.view(orig_shape)

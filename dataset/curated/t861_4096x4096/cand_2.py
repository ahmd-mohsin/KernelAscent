import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 861
M, D, DT = 4096, 4096, torch.float16


@triton.jit
def _ln_relu_kernel(
    X, G, B, Y,
    stride_x, stride_y,
    N, eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = (x - mean) * rstd * g + b
    y = tl.maximum(y, 0.0)

    tl.store(Y + row * stride_y + cols, y.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        if not x.is_cuda:
            x = F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)
            return torch.relu(x)

        x = x.contiguous()
        M_, N = x.shape
        y = torch.empty_like(x)
        BLOCK_N = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK_N >= 1024 else 4
        _ln_relu_kernel[(M_,)](
            x, self.ln1_g, self.ln1_b, y,
            x.stride(0), y.stride(0),
            N, 1e-5,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return y

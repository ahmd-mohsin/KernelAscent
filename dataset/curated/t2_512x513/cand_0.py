import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 2
M, D, DT = 512, 513, torch.float16


@triton.jit
def _fused_post_kernel(
    X_ptr, W_ptr, Y_ptr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * N + cols, mask=mask, other=-float('inf')).to(tl.float32)

    # softmax 1 (fp32 compute, fp16 output like PyTorch)
    m = tl.max(x, 0)
    e = tl.exp(x - m)
    s = tl.sum(tl.where(mask, e, 0.0), 0)
    x = (e / s).to(tl.float16).to(tl.float32)

    # exact GELU (erf-based), fp32 compute, round to fp16
    x = (0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))).to(tl.float16).to(tl.float32)

    # RMSNorm in fp32, cast to fp16, multiply by weight
    xm = tl.where(mask, x, 0.0)
    ms = tl.sum(xm * xm, 0) / N
    inv = tl.math.rsqrt(ms + 1e-6)
    x = (x * inv).to(tl.float16).to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x = (x * w).to(tl.float16).to(tl.float32)

    # softmax 2
    x = tl.where(mask, x, -float('inf'))
    m2 = tl.max(x, 0)
    e2 = tl.exp(x - m2)
    s2 = tl.sum(tl.where(mask, e2, 0.0), 0)
    y = (e2 / s2).to(tl.float16)

    tl.store(Y_ptr + row * N + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 1024, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback: reference implementation
            x = x @ self.W0
            x = torch.softmax(x, dim=-1)
            x = F.gelu(x)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms3_w
            x = torch.softmax(x, dim=-1)
            return x

        # cuBLAS matmul (tensor cores)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        rows, N = h.shape
        out = torch.empty_like(h)

        _fused_post_kernel[(rows,)](
            h, self.rms3_w, out,
            N=N,
            BLOCK=triton.next_power_of_2(N),
            num_warps=8,
        )
        return out

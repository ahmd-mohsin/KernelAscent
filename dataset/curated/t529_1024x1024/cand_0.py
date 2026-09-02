import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 529
M, D, DT = 1024, 1024, torch.bfloat16


@triton.jit
def _fused_bias_rms_kernel(
    X, B, W, OUT,
    N, stride_x, stride_o,
    EPS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)

    # bf16 add with fp32 opmath, round back to bf16 (matches PyTorch)
    s = (x.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)
    sf = s.to(tl.float32)

    ms = tl.sum(tl.where(mask, sf * sf, 0.0), axis=0) / N
    inv = 1.0 / tl.sqrt(ms + EPS)

    y = (sf * inv).to(tl.bfloat16)

    w = tl.load(W + cols, mask=mask, other=0.0)
    z = (y.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)
    out = (z.to(tl.float32) * SCALE).to(tl.bfloat16)

    tl.store(OUT + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 4096, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 matmul (tensor cores)
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _fused_bias_rms_kernel[(m,)](
            h, self.b1, self.rms2_w, out,
            n, h.stride(0), out.stride(0),
            EPS=1e-6, SCALE=1.181, BLOCK=BLOCK,
            num_warps=8,
        )
        return out

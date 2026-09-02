import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 542
M, D, DT = 4096, 513, torch.float16


@triton.jit
def _softmax_bias_ln_kernel(
    X, B2, G, B, Y,
    N, stride_x, stride_y,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=-float('inf')).to(tl.float32)

    # softmax (fp32 accumulation, like PyTorch on half)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = e / s

    # cast to fp16, add bias in fp16 (matches reference rounding)
    p16 = p.to(tl.float16)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0)
    h16 = p16 + b2
    h = h16.to(tl.float32)

    # layernorm in fp32
    Nf = N.to(tl.float32)
    mean = tl.sum(tl.where(mask, h, 0.0), axis=0) / Nf
    d = tl.where(mask, h - mean, 0.0)
    var = tl.sum(d * d, axis=0) / Nf
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g + b

    tl.store(Y + row * stride_y + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 2048, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS tensor-core GEMM
        if not h.is_cuda:
            h = torch.softmax(h, dim=-1) + self.b2
            return F.layer_norm(h, (h.shape[-1],), self.ln3_g, self.ln3_b)

        h = h.contiguous()
        Mrows, N = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _softmax_bias_ln_kernel[(Mrows,)](
            h, self.b2, self.ln3_g, self.ln3_b, y,
            N, h.stride(0), y.stride(0),
            1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y

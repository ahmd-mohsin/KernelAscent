import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 743
M, D, DT = 2048, 513, torch.bfloat16


@triton.jit
def _relu_softmax_scale_ln_kernel(
    X_ptr, G_ptr, B_ptr, Y_ptr,
    N,
    EPS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * N + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # ReLU
    x = tl.maximum(x, 0.0)

    # Softmax in fp32 (as torch does internally for bf16)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = e / s

    # cast back to bf16 (softmax output), then scalar mul (opmath fp32 -> bf16)
    p = p.to(tl.bfloat16).to(tl.float32)
    p = p * SCALE
    p = p.to(tl.bfloat16).to(tl.float32)

    # LayerNorm in fp32
    mean = tl.sum(p, axis=0) / N
    d = tl.where(mask, p - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    g = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    y = d * rstd * g + b
    tl.store(Y_ptr + row * N + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 2048, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS bf16 matmul (tensor cores on A100)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()

        rows, N = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _relu_softmax_scale_ln_kernel[(rows,)](
            h, self.ln4_g, self.ln4_b, out,
            N,
            EPS=1e-5,
            SCALE=1.1424,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out

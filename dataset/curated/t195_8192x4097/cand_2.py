import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 195
M, D, DT = 8192, 4097, torch.float16


@triton.jit
def _softmax_ln_scale_kernel(
    X_ptr, G_ptr, B_ptr, Y_ptr,
    N, scale, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # load row (fp16 -> fp32 compute, matching PyTorch's acc type for half)
    x = tl.load(X_ptr + row * N + offs, mask=mask, other=-float('inf')).to(tl.float32)

    # ---- softmax (fp32 accumulation, output rounded to fp16 like PyTorch) ----
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = e / s
    # PyTorch materializes softmax output in fp16 before layer_norm reads it
    p = p.to(tl.float16).to(tl.float32)

    # ---- layer norm (mean/var in fp32, biased variance, eps=1e-5) ----
    mean = tl.sum(tl.where(mask, p, 0.0), axis=0) / N
    d = tl.where(mask, p - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    y = d * rstd * g + b
    # PyTorch layer_norm writes fp16, then multiplies by scalar (double rounding)
    y = y.to(tl.float16).to(tl.float32) * scale

    tl.store(Y_ptr + row * N + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 2048, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS tensor cores
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        rows, N = h.shape[-2] if h.dim() > 1 else 1, h.shape[-1]
        h2 = h.view(-1, N)
        y = torch.empty_like(h2)
        BLOCK = triton.next_power_of_2(N)
        _softmax_ln_scale_kernel[(h2.shape[0],)](
            h2, self.ln2_g, self.ln2_b, y,
            N, 1.0319, 1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y.view(h.shape)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 666
M, D, DT = 4096, 4096, torch.float16


@triton.jit
def _fused_ln_softmax_add_softmax(
    X, G, B, B3, Y,
    N: tl.constexpr,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x_ptr = X + row * N + cols

    # ---- LayerNorm (fp32 accumulate, output rounded to fp16 like PyTorch) ----
    x = tl.load(x_ptr, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    xm = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xm * xm, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = xm * rstd * g + b
    # round to fp16 (intermediate tensor dtype in reference)
    y = y.to(tl.float16).to(tl.float32)

    # ---- Softmax 1 (fp32 accumulate, rounded to fp16) ----
    y_masked = tl.where(mask, y, float("-inf"))
    mx = tl.max(y_masked, axis=0)
    e = tl.exp(y_masked - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y16 = (e / s).to(tl.float16)

    # ---- Add bias (fp16 arithmetic, as in reference) ----
    b3 = tl.load(B3 + cols, mask=mask, other=0.0)
    y16 = y16 + b3  # fp16 add

    # ---- Softmax 2 (fp32 accumulate, rounded to fp16) ----
    y2 = y16.to(tl.float32)
    y2_masked = tl.where(mask, y2, float("-inf"))
    mx2 = tl.max(y2_masked, axis=0)
    e2 = tl.exp(y2_masked - mx2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    out = (e2 / s2).to(tl.float16)

    tl.store(Y + row * N + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores on A100)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 1024 else 4
        _fused_ln_softmax_add_softmax[(m,)](
            h, self.ln1_g, self.ln1_b, self.b3, out,
            N=n, eps=1e-5, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 437
M, D, DT = 8192, 2049, torch.float16


@triton.jit
def _fused_ln_softmax_ln(
    X, Y, G1, B1, G3, B3,
    N, stride_x, stride_y,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm 1 (fp32 compute, matching PyTorch half accumulation) ----
    n = N.to(tl.float32)
    mean = tl.sum(x, axis=0) / n
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / n
    rstd = 1.0 / tl.sqrt(var + eps)
    g1 = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    y = diff * rstd * g1 + b1
    # round to fp16 (intermediate tensor dtype in reference) then back to fp32
    y = y.to(tl.float16).to(tl.float32)

    # ---- Softmax (fp32 compute) ----
    y_masked = tl.where(mask, y, float("-inf"))
    mx = tl.max(y_masked, axis=0)
    e = tl.exp(y_masked - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = e / s
    p = p.to(tl.float16).to(tl.float32)

    # ---- LayerNorm 2 ----
    mean2 = tl.sum(p, axis=0) / n
    diff2 = tl.where(mask, p - mean2, 0.0)
    var2 = tl.sum(diff2 * diff2, axis=0) / n
    rstd2 = 1.0 / tl.sqrt(var2 + eps)
    g3 = tl.load(G3 + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    out = diff2 * rstd2 * g3 + b3

    tl.store(Y + row * stride_y + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 2048, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS GEMM (tensor cores)
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_ln_softmax_ln[(Mrows,)](
            h, out,
            self.ln1_g, self.ln1_b, self.ln3_g, self.ln3_b,
            N, h.stride(0), out.stride(0),
            1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out

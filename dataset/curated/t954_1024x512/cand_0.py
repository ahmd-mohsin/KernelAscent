import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 954
M, D, DT = 1024, 512, torch.float16


@triton.jit
def _fused_scale_ln_softmax_relu(
    X, G, B, Y,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    # scale by 1.0315 (fp32 opmath, rounded back to fp16, matching PyTorch)
    x = (x * 1.0315).to(tl.float16).to(tl.float32)

    # layer norm (fp32 accumulation)
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (d * rstd * g + b).to(tl.float16).to(tl.float32)

    # softmax (fp32)
    y_masked = tl.where(mask, y, float('-inf'))
    mx = tl.max(y_masked, axis=0)
    e = tl.where(mask, tl.exp(y_masked - mx), 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    # relu (no-op on softmax output, kept for equivalence)
    out = tl.maximum(out, 0.0)

    tl.store(Y + row * N + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS fp16 GEMM (tensor cores), identical to reference matmul
        h = x @ self.W0
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_scale_ln_softmax_relu[(Mrows,)](
            h, self.ln2_g, self.ln2_b, out,
            N, 1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out

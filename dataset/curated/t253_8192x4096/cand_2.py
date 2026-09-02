import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 253
M, D, DT = 8192, 4096, torch.float16


@triton.jit
def _fused_softmax_ln_bias(
    X, G, B, B3, Y,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + row * N + offs, mask=mask, other=-float('inf')).to(tl.float32)

    # softmax (fp32 compute, round to fp16 like PyTorch inter-op storage)
    mx = tl.max(x, 0)
    e = tl.exp(x - mx)
    s = tl.sum(tl.where(mask, e, 0.0), 0)
    sm = (e / s).to(tl.float16).to(tl.float32)
    sm = tl.where(mask, sm, 0.0)

    # layer norm (fp32 compute)
    mean = tl.sum(sm, 0) / N
    diff = tl.where(mask, sm - mean, 0.0)
    var = tl.sum(diff * diff, 0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    y = (diff * rstd * g + b).to(tl.float16).to(tl.float32)

    # bias add (opmath fp32, output fp16 like PyTorch)
    b3 = tl.load(B3 + offs, mask=mask, other=0.0).to(tl.float32)
    out = (y + b3).to(tl.float16)
    tl.store(Y + row * N + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 4096, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = torch.matmul(x, self.W0)  # cuBLAS fp16 GEMM (tensor cores)
        h = h.contiguous()
        Mrows, N = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_softmax_ln_bias[(Mrows,)](
            h, self.ln2_g, self.ln2_b, self.b3, y,
            N, 1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y

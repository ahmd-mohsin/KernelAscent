import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 508
M, D, DT = 8192, 4096, torch.float16


@triton.jit
def _fused_softmax_gelu_ln_bias(
    X, G, B, B4, Y,
    N, stride_x, stride_y,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=-float('inf')).to(tl.float32)

    # softmax (fp32 accumulation, output rounded to fp16 like PyTorch)
    m = tl.max(x, 0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, 0)
    p = (e / s).to(tl.float16).to(tl.float32)

    # exact (erf) GELU, fp32 compute, rounded to fp16 like PyTorch
    g = 0.5 * p * (1.0 + tl.math.erf(p * 0.7071067811865476))
    g = tl.where(mask, g, 0.0).to(tl.float16).to(tl.float32)

    # layernorm (fp32 accumulation)
    mean = tl.sum(g, 0) / N
    d = tl.where(mask, g - mean, 0.0)
    var = tl.sum(d * d, 0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    w = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    bb = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * w + bb
    y = y.to(tl.float16).to(tl.float32)

    # bias add
    b4 = tl.load(B4 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (y + b4).to(tl.float16)

    tl.store(Y + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores)
        h = torch.matmul(x, self.W0)
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _fused_softmax_gelu_ln_bias[(m,)](
            h, self.ln3_g, self.ln3_b, self.b4, out,
            n, h.stride(0), out.stride(0),
            EPS=1e-5,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out

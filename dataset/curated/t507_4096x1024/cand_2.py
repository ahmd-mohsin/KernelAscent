import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 507
M, D, DT = 4096, 1024, torch.float16


@triton.jit
def _fused_ln_gelu_softmax(
    X_ptr, OUT_ptr, G_ptr, B_ptr, B3_ptr,
    N, stride,
    eps, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride + offs, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm (fp32 math, like PyTorch's fp16 layer_norm)
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g + b
    # round to fp16 (stage boundary in reference)
    y = y.to(tl.float16).to(tl.float32)

    # scale
    y = y * scale
    y = y.to(tl.float16).to(tl.float32)

    # bias
    b3 = tl.load(B3_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = y + b3
    y = y.to(tl.float16).to(tl.float32)

    # exact GELU
    y = 0.5 * y * (1.0 + tl.math.erf(y * 0.7071067811865476))
    y = y.to(tl.float16).to(tl.float32)

    # softmax (fp32 accumulation)
    y = tl.where(mask, y, float('-inf'))
    m = tl.max(y, axis=0)
    e = tl.exp(y - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(OUT_ptr + row * stride + offs, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        _fused_ln_gelu_softmax[(Mrows,)](
            h, out, self.ln1_g, self.ln1_b, self.b3,
            N, h.stride(0),
            1e-5, 1.2202,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out

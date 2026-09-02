import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 353
M, D, DT = 512, 512, torch.float16


@triton.jit
def _fused_gelu_ln_softmax_kernel(
    X, G, B, B5, Y,
    stride_xm, stride_ym,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)

    # exact GELU (erf), round through fp16 to match PyTorch half GELU output
    inv_sqrt2: tl.constexpr = 0.7071067811865476
    g = 0.5 * x * (1.0 + tl.math.erf(x * inv_sqrt2))
    g = g.to(tl.float16).to(tl.float32)

    # LayerNorm (fp32 stats, fp16 output like PyTorch)
    mean = tl.sum(tl.where(mask, g, 0.0), axis=0) / N
    diff = tl.where(mask, g - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    gamma = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    h = (g - mean) * rstd * gamma + beta
    h = h.to(tl.float16).to(tl.float32)

    # Softmax (fp32 compute, fp16 output like PyTorch)
    h_masked = tl.where(mask, h, float('-inf'))
    m = tl.max(h_masked, axis=0)
    e = tl.exp(h_masked - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = (e / s).to(tl.float16).to(tl.float32)

    # scale (fp16 rounding) then add bias (fp16 rounding)
    z = (sm * 1.2464).to(tl.float16).to(tl.float32)
    b5 = tl.load(B5 + cols, mask=mask, other=0.0).to(tl.float32)
    out = (z + b5).to(tl.float16)

    tl.store(Y + row * stride_ym + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b5 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS tensor-core GEMM
        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _fused_gelu_ln_softmax_kernel[(m,)](
            x, self.ln2_g, self.ln2_b, self.b5, y,
            x.stride(0), y.stride(0),
            N=n, BLOCK=BLOCK,
            num_warps=4,
        )
        return y

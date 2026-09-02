import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 772
M, D, DT = 512, 2048, torch.float16


@triton.jit
def _ln_relu_softmax_kernel(
    X, W, B, Y,
    stride_xm, stride_ym,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm (fp32 accumulation, matching PyTorch half layer_norm)
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = (x - mean) * rstd * w + b
    # cast to fp16 (layer_norm output dtype), then relu in fp16
    y16 = y.to(tl.float16)
    r16 = tl.maximum(y16, 0.0)

    # softmax in fp32 (matching PyTorch half softmax upcast)
    r = r16.to(tl.float32)
    r = tl.where(mask, r, float('-inf'))
    m = tl.max(r, axis=0)
    e = tl.exp(r - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y + row * stride_ym + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS tensor-core GEMM
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _ln_relu_softmax_kernel[(Mrows,)](
            h, self.ln1_g, self.ln1_b, out,
            h.stride(0), out.stride(0),
            N, 1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out

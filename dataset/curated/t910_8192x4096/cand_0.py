import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 910
M, D, DT = 8192, 4096, torch.float16


@triton.jit
def _fused_relu_bias_relu_ln(
    X, B, G, Beta, Out,
    N, eps,
    stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    # relu -> +bias -> relu
    x = tl.maximum(x, 0.0)
    x = x + b
    x = tl.maximum(x, 0.0)

    # layernorm in fp32 (matches PyTorch fp16 layer_norm internals)
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(Beta + cols, mask=mask, other=0.0).to(tl.float32)

    y = (x - mean) * rstd * g + beta
    tl.store(Out + row * stride_o + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 4096, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS GEMM (tensor cores)
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_relu_bias_relu_ln[(m,)](
            h, self.b2, self.ln4_g, self.ln4_b, out,
            n, 1e-5,
            h.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out

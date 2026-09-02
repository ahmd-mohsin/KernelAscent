import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 534
M, D, DT = 4096, 1024, torch.float16


@triton.jit
def _fused_bias_softmax_scale_ln(
    X_ptr, Bias_ptr, G_ptr, Beta_ptr, Y_ptr,
    stride_x, stride_y,
    N, SCALE, EPS,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=float('-inf')).to(tl.float32)
    b = tl.load(Bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x = x + b

    # softmax (fp32 accumulation, matching PyTorch's fp16 softmax internals)
    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    p = e / denom

    # mimic fp16 rounding of softmax output, then scale in fp16 like `x * 1.0871`
    p = p.to(tl.float16).to(tl.float32)
    p = p * SCALE
    p = p.to(tl.float16).to(tl.float32)

    # layer norm (fp32 statistics, matching PyTorch layer_norm on fp16)
    mean = tl.sum(p, axis=0) / N
    diff = tl.where(mask, p - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    g = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(Beta_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = diff * rstd * g + beta

    tl.store(Y_ptr + row * stride_y + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 4096, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS GEMM (tensor cores)
        h = x @ self.W0
        h = h.contiguous()

        rows, N = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        _fused_bias_softmax_scale_ln[(rows,)](
            h, self.b1, self.ln4_g, self.ln4_b, out,
            h.stride(0), out.stride(0),
            N, 1.0871, 1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 266
M, D, DT = 512, 512, torch.float16


@triton.jit
def _fused_bias_relu_scale_ln(
    X, B1, B2, G, B, Y,
    stride_x, stride_y,
    N, eps, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0)

    # x + b1 (fp32 add then round to fp16 == fp16 add, matches PyTorch)
    t = (x.to(tl.float32) + b1.to(tl.float32)).to(tl.float16)
    # + b2
    t = (t.to(tl.float32) + b2.to(tl.float32)).to(tl.float16)
    # relu
    t = tl.maximum(t, 0.0)
    # * scale (opmath fp32, round to fp16)
    t = (t.to(tl.float32) * scale).to(tl.float16)

    # layernorm in fp32 (matches PyTorch half layer_norm internals)
    tf = t.to(tl.float32)
    tf = tl.where(mask, tf, 0.0)
    mean = tl.sum(tf, axis=0) / N
    diff = tl.where(mask, tf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (tf - mean) * rstd * g + b

    tl.store(Y + row * stride_y + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln5_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln5_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS matmul (identical to reference)
        h = h.contiguous()
        m, n = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _fused_bias_relu_scale_ln[(m,)](
            h, self.b1, self.b2, self.ln5_g, self.ln5_b, y,
            h.stride(0), y.stride(0),
            n, 1e-5, 1.0466,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return y

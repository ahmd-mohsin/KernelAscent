import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.language.extra.libdevice as libdevice

SEED = 1
M, D, DT = 1024, 513, torch.bfloat16


@triton.jit
def _fused_bias_rms_softmax_bias_relu(
    X_ptr, B1_ptr, W_ptr, B4_ptr, Out_ptr,
    stride_x, stride_o,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)

    # load row (bf16) and bias, add (compute in fp32, round to bf16 - matches PyTorch opmath)
    x = tl.load(X_ptr + row * stride_x + offs)
    b1 = tl.load(B1_ptr + offs)
    xb = (x.to(tl.float32) + b1.to(tl.float32)).to(tl.bfloat16)

    # RMSNorm in fp32
    xf = xb.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    r = libdevice.rsqrt(ms + 1e-6)

    w = tl.load(W_ptr + offs)
    y = ((xf * r).to(tl.bfloat16).to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    # softmax in fp32 (matches PyTorch bf16 softmax accumulation)
    yf = y.to(tl.float32)
    mx = tl.max(yf, axis=0)
    e = libdevice.exp(yf - mx)
    s = tl.sum(e, axis=0)
    p = (e / s).to(tl.bfloat16)

    # + b4, relu
    b4 = tl.load(B4_ptr + offs)
    o = (p.to(tl.float32) + b4.to(tl.float32)).to(tl.bfloat16)
    o = tl.maximum(o, 0.0)

    tl.store(Out_ptr + row * stride_o + offs, o)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 4096, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS (tensor cores)
        x = x @ self.W0
        x = x.contiguous()
        m, n = x.shape
        out = torch.empty_like(x)
        grid = (m,)
        _fused_bias_rms_softmax_bias_relu[grid](
            x, self.b1, self.rms2_w, self.b4, out,
            x.stride(0), out.stride(0),
            N=n, BLOCK=4096,
            num_warps=8,
        )
        return out

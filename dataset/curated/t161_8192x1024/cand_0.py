import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 161
M, D, DT = 8192, 1024, torch.float16


@triton.jit
def _fused_bias_softmax2_rms_kernel(
    X_ptr, B1_ptr, B2_ptr, W_ptr, Y_ptr,
    stride_x, stride_y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)

    x = tl.load(X_ptr + row * stride_x + offs)          # fp16
    b1 = tl.load(B1_ptr + offs)                          # fp16
    b2 = tl.load(B2_ptr + offs)                          # fp16

    # bias adds in fp16 (matches eager x + b1 then + b2)
    x = x + b1
    x = x + b2

    # softmax #1: fp16 input, fp32 math, fp16 output (matches PyTorch)
    xf = x.to(tl.float32)
    mx = tl.max(xf, axis=0)
    e = tl.exp(xf - mx)
    s = tl.sum(e, axis=0)
    x = (e / s).to(tl.float16)

    # softmax #2
    xf = x.to(tl.float32)
    mx = tl.max(xf, axis=0)
    e = tl.exp(xf - mx)
    s = tl.sum(e, axis=0)
    x = (e / s).to(tl.float16)

    # RMSNorm: fp32 mean-of-squares, cast to fp16, then fp16 multiply by weight
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    y = (xf * r).to(tl.float16)

    w = tl.load(W_ptr + offs)                            # fp16
    y = y * w

    tl.store(Y_ptr + row * stride_y + offs, y)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 4096, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms5_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores on A100)
        h = x @ self.W0
        if not h.is_contiguous():
            h = h.contiguous()

        m, n = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(n)
        _fused_bias_softmax2_rms_kernel[(m,)](
            h, self.b1, self.b2, self.rms5_w, out,
            h.stride(0), out.stride(0),
            N=n,
            BLOCK=BLOCK,
            num_warps=8,
            num_stages=1,
        )
        return out

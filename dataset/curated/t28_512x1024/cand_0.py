import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 28
M, D, DT = 512, 1024, torch.float16


@triton.jit
def _fused_scale_rms_softmax(
    X_ptr, W_ptr, Out_ptr,
    stride_x, stride_o,
    N,
    SCALE: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # load matmul output (fp16) and upcast
    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # x = x * 1.1067  (fp16 semantics: compute in f32, round to fp16 -> matches PyTorch opmath)
    y16 = (x * SCALE).to(tl.float16)
    y = y16.to(tl.float32)

    # RMS norm in fp32
    ms = tl.sum(tl.where(mask, y * y, 0.0), axis=0) / N
    r = tl.math.rsqrt(ms + EPS)

    # (y * r) rounded to fp16, then multiplied by weight (fp16 semantics)
    n16 = (y * r).to(tl.float16)
    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    z16 = (n16.to(tl.float32) * w).to(tl.float16)
    z = z16.to(tl.float32)

    # softmax in fp32 (matches PyTorch acc type for half input)
    z = tl.where(mask, z, float('-inf'))
    zmax = tl.max(z, axis=0)
    e = tl.exp(z - zmax)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(Out_ptr + row * stride_o + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 4096, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS tensor-core GEMM
        h = torch.matmul(x, self.W0)
        if not h.is_contiguous():
            h = h.contiguous()

        Mrows, N = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        _fused_scale_rms_softmax[(Mrows,)](
            h, self.rms2_w, out,
            h.stride(0), out.stride(0),
            N,
            SCALE=1.1067,
            EPS=1e-6,
            BLOCK=BLOCK,
            num_warps=16,
        )
        return out

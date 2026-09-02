import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 101
M, D, DT = 8192, 1024, torch.bfloat16


@triton.jit
def _relu_rms_kernel(X, W, Y, N, stride, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride + cols, mask=mask, other=0.0)
    # relu in input dtype (bf16), then cast to fp32 (matches x.float() after relu)
    x = tl.where(x > 0, x, x * 0)
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    r = 1.0 / tl.sqrt(ms + eps)
    normed = (xf * r).to(tl.bfloat16)  # round to bf16 first (matches .to(x.dtype))
    w = tl.load(W + cols, mask=mask, other=0.0)
    out = (normed.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)
    tl.store(Y + row * stride + cols, out, mask=mask)


@triton.jit
def _relu_scale_kernel(X, Y, n_elements, scale, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(X + offs, mask=mask, other=0.0)
    xf = x.to(tl.float32)
    xf = tl.where(xf > 0, xf, 0.0)
    out = (xf * scale).to(tl.bfloat16)
    tl.store(Y + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 1024, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.W3 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS bf16 GEMM (tensor cores)
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _relu_rms_kernel[(m,)](
            x, self.rms2_w, y, n, x.stride(0), 1e-6,
            BLOCK=BLOCK, num_warps=8,
        )
        z = y @ self.W3
        out = torch.empty_like(z)
        n_el = z.numel()
        grid = (triton.cdiv(n_el, 2048),)
        _relu_scale_kernel[grid](z, out, n_el, 1.0805, BLOCK=2048, num_warps=8)
        return out

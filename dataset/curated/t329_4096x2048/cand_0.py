import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 329
M, D, DT = 4096, 2048, torch.float16


@triton.jit
def _rms_gelu_kernel(X, W, Y, N, stride_x, stride_y, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    # RMS norm in fp32
    ms = tl.sum(x * x, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + EPS)
    xn = (x * inv).to(tl.float16)  # round to fp16 like reference .to(x.dtype)
    w = tl.load(W + cols, mask=mask, other=0.0)
    # fp16*fp16 elementwise == fp32 mult then round (product exact in fp32)
    h = (xn.to(tl.float32) * w.to(tl.float32)).to(tl.float16)
    # GELU (erf), computed in fp32 (PyTorch opmath), rounded to fp16
    hf = h.to(tl.float32)
    g = (0.5 * hf * (1.0 + tl.math.erf(hf * 0.7071067811865476))).to(tl.float16)
    # scale by 1.203 in fp32 opmath, round to fp16
    out = (g.to(tl.float32) * 1.203).to(tl.float16)
    tl.store(Y + row * stride_y + cols, out, mask=mask)


@triton.jit
def _relu_kernel(X, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(X + offs, mask=mask)
    tl.store(X + offs, tl.maximum(x, 0.0), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.W4 = nn.Parameter((torch.randn(512, 1024, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _rms_gelu_kernel[(m,)](
            x, self.rms1_w, y, n, x.stride(0), y.stride(0),
            EPS=1e-6, BLOCK=BLOCK, num_warps=4,
        )
        out = y @ self.W4
        n_el = out.numel()
        _relu_kernel[(triton.cdiv(n_el, 4096),)](out, n_el, BLOCK=4096, num_warps=8)
        return out

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 193
M, D, DT = 8192, 4096, torch.float16


@triton.jit
def _rms_softmax_kernel(
    X, W, Y,
    stride_x, stride_y,
    N: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x16 = tl.load(X + row * stride_x + offs, mask=mask, other=0.0)
    xf = x16.to(tl.float32)

    # RMSNorm in fp32, matching: rsqrt(mean(x^2) + eps)
    ms = tl.sum(xf * xf, axis=0) / N
    r = 1.0 / tl.sqrt(ms + EPS)

    # cast normalized value back to fp16, then multiply by fp16 weight (fp16 math),
    # exactly matching:  (xf * rsqrt(...)).to(fp16) * w
    xn = (xf * r).to(tl.float16)
    w = tl.load(W + offs, mask=mask, other=0.0)
    v = xn * w  # fp16

    # softmax with fp32 accumulation (matches PyTorch half softmax semantics)
    vf = v.to(tl.float32)
    vf = tl.where(mask, vf, float('-inf'))
    mx = tl.max(vf, axis=0)
    e = tl.exp(vf - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(Y + row * stride_y + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W1 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback (reference path)
            x = x * 1.0122
            x = x @ self.W1
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            return torch.softmax(x, dim=-1)

        # scale (same elementwise fp16 op as reference), then cuBLAS GEMM
        x = x * 1.0122
        h = torch.matmul(x, self.W1)
        h = h.contiguous()

        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _rms_softmax_kernel[(Mrows,)](
            h, self.rms2_w, out,
            h.stride(0), out.stride(0),
            N=N, EPS=1e-6, BLOCK=BLOCK,
            num_warps=8,
        )
        return out

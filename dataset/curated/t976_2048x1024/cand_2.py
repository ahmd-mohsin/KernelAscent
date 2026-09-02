import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 976
M, D, DT = 2048, 1024, torch.float16


@triton.jit
def _fused_double_rmsnorm_kernel(
    X, W0, W2, Y,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    eps,
    scale,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * D + offs, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # First RMSNorm (compute in fp32, round to fp16, elementwise mul in fp32->fp16
    # which matches PyTorch's opmath behavior for Half tensors)
    ms = tl.sum(xf * xf, axis=0) / D
    r = tl.math.rsqrt(ms + eps)
    w0 = tl.load(W0 + offs, mask=mask, other=0.0)
    x1 = ((xf * r).to(tl.float16).to(tl.float32) * w0.to(tl.float32)).to(tl.float16)

    # Scale by constant (fp32 opmath, round back to fp16 like PyTorch)
    x2 = (x1.to(tl.float32) * scale).to(tl.float16)

    # Second RMSNorm
    x2f = x2.to(tl.float32)
    ms2 = tl.sum(x2f * x2f, axis=0) / D
    r2 = tl.math.rsqrt(ms2 + eps)
    w2 = tl.load(W2 + offs, mask=mask, other=0.0)
    y = ((x2f * r2).to(tl.float16).to(tl.float32) * w2.to(tl.float32)).to(tl.float16)

    tl.store(Y + row * D + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.W3 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # Reference fallback for CPU tensors
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            x = x * 1.4509
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            return x @ self.W3

        x = x.contiguous()
        rows, d = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        _fused_double_rmsnorm_kernel[(rows,)](
            x, self.rms0_w, self.rms2_w, y,
            d, BLOCK,
            1e-6, 1.4509,
            num_warps=8,
        )
        # Final projection via cuBLAS (tensor cores, fp16)
        return y @ self.W3

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 881
M, D, DT = 2048, 1024, torch.float16


@triton.jit
def _fused_row_kernel(
    x_ptr, w_ptr, out_ptr,
    D,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    base = row * D

    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)

    # x = x * 1.1029  (half output, fp32 compute)
    x = (x * 1.1029).to(tl.float16).to(tl.float32)

    # softmax 1 (fp32 compute, round to fp16)
    m = tl.max(tl.where(mask, x, float('-inf')), 0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, 0)
    x = (e / s).to(tl.float16).to(tl.float32)

    # RMSNorm in fp32, cast to fp16, then multiply by fp16 weight (fp32 compute)
    ms = tl.sum(x * x, 0) / D
    r = 1.0 / tl.sqrt(ms + 1e-6)
    xn = (x * r).to(tl.float16).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x = (xn * w).to(tl.float16).to(tl.float32)

    # softmax 2
    m = tl.max(tl.where(mask, x, float('-inf')), 0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, 0)
    x = (e / s).to(tl.float16).to(tl.float32)

    # softmax 3
    m = tl.max(tl.where(mask, x, float('-inf')), 0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, 0)
    x = (e / s).to(tl.float16)

    tl.store(out_ptr + base + offs, x, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            return self._forward_ref(x)
        orig_shape = x.shape
        d = orig_shape[-1]
        xc = x.contiguous().view(-1, d)
        rows = xc.shape[0]
        out = torch.empty_like(xc)
        BLOCK = triton.next_power_of_2(d)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_row_kernel[(rows,)](
            xc, self.rms2_w, out, d,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.view(orig_shape)

    def _forward_ref(self, x):
        x = x * 1.1029
        x = torch.softmax(x, dim=-1)
        _xf = x.float()
        x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
        x = torch.softmax(x, dim=-1)
        x = torch.softmax(x, dim=-1)
        return x

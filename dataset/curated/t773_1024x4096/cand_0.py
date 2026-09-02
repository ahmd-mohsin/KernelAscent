import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

try:
    from triton.language.extra import libdevice as _ld
    _HAS_LD = True
except Exception:
    _HAS_LD = False

SEED = 773
M, D, DT = 1024, 4096, torch.bfloat16


@triton.jit
def _fused_kernel(X, B, Y, n_cols, stride_x, stride_y, BLOCK: tl.constexpr, USE_LD: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(X + row * stride_x + cols, mask=mask, other=-float('inf')).to(tl.float32)

    # ---- softmax 1 (fp32 accumulation, round to bf16 like PyTorch) ----
    m1 = tl.max(x, axis=0)
    e1 = x - m1
    if USE_LD:
        e1 = _ld.exp(e1)
    else:
        e1 = tl.exp(e1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, axis=0)
    y1 = (e1 / s1).to(tl.bfloat16)

    # ---- softmax 2 ----
    x2 = tl.where(mask, y1.to(tl.float32), -float('inf'))
    m2 = tl.max(x2, axis=0)
    e2 = x2 - m2
    if USE_LD:
        e2 = _ld.exp(e2)
    else:
        e2 = tl.exp(e2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    y2 = (e2 / s2).to(tl.bfloat16)

    # ---- relu -> +b3 -> *1.3071 (each op in fp32 then rounded to bf16, matching PyTorch) ----
    r = tl.maximum(y2.to(tl.float32), 0.0).to(tl.bfloat16)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    a = (r.to(tl.float32) + b).to(tl.bfloat16)
    out = (a.to(tl.float32) * 1.3071).to(tl.bfloat16)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b3 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = torch.softmax(x, dim=-1)
            y = torch.softmax(y, dim=-1)
            y = torch.relu(y)
            y = y + self.b3
            return y * 1.3071

        orig_shape = x.shape
        n_cols = orig_shape[-1]
        x2d = x.reshape(-1, n_cols)
        if not x2d.is_contiguous():
            x2d = x2d.contiguous()
        n_rows = x2d.shape[0]

        out = torch.empty_like(x2d)
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_kernel[(n_rows,)](
            x2d, self.b3, out,
            n_cols, x2d.stride(0), out.stride(0),
            BLOCK=BLOCK, USE_LD=_HAS_LD,
            num_warps=num_warps,
        )
        return out.reshape(orig_shape)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 880
M, D, DT = 512, 512, torch.float16


@triton.jit
def _fused_kernel(x_ptr, b_ptr, w_ptr, out_ptr, N, stride_x, stride_o,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # bias add (match fp16 semantics of x + b0)
    xb = (x + b).to(tl.float16).to(tl.float32)
    xb = tl.where(mask, xb, float('-inf'))

    # softmax
    m = tl.max(xb, axis=0)
    e = tl.exp(xb - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = (e / s).to(tl.float16)

    # rmsnorm in fp32 on the fp16 softmax result
    smf = sm.to(tl.float32)
    ms = tl.sum(smf * smf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    y = (smf * inv).to(tl.float16)

    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    out = y * w

    tl.store(out_ptr + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        Mrows, N = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_kernel[(Mrows,)](
            x, self.b0, self.rms2_w, out,
            N, x.stride(0), out.stride(0),
            BLOCK=BLOCK, num_warps=4,
        )
        return out

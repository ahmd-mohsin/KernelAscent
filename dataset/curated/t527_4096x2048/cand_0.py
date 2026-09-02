import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 527
M, D, DT = 4096, 2048, torch.float16


@triton.jit
def _fused_double_rmsnorm_relu(
    X, W0, W1, Y,
    stride_x, stride_y,
    D_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D_

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)  # fp16
    w0 = tl.load(W0 + cols, mask=mask, other=0.0)                 # fp16
    w1 = tl.load(W1 + cols, mask=mask, other=0.0)                 # fp16

    # --- RMSNorm 1 (compute in fp32, cast back to fp16, mul by w in fp16) ---
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / D_
    rs = 1.0 / tl.sqrt(ms + 1e-6)
    h = (xf * rs).to(tl.float16) * w0
    # ReLU
    h = tl.maximum(h, tl.zeros_like(h))

    # --- RMSNorm 2 ---
    hf = h.to(tl.float32)
    hf = tl.where(mask, hf, 0.0)
    ms2 = tl.sum(hf * hf, axis=0) / D_
    rs2 = 1.0 / tl.sqrt(ms2 + 1e-6)
    o = (hf * rs2).to(tl.float16) * w1
    o = tl.maximum(o, tl.zeros_like(o))

    tl.store(Y + row * stride_y + cols, o, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        assert x.is_cuda and x.dtype == torch.float16
        x = x.contiguous()
        M_, D_ = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(D_)
        _fused_double_rmsnorm_relu[(M_,)](
            x, self.rms0_w, self.rms2_w, y,
            x.stride(0), y.stride(0),
            D_=D_, BLOCK=BLOCK,
            num_warps=8,
        )
        return y

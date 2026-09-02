import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 190
M, D, DT = 512, 1024, torch.bfloat16


@triton.jit
def _fused_scale_rms_bias(
    X_ptr, W_ptr, B_ptr, Out_ptr,
    N, stride_x, stride_o,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0)  # bf16
    # x * 1.0427 computed in fp32 then rounded to bf16 (matches PyTorch semantics)
    xf = x.to(tl.float32) * 1.0427
    x_bf = xf.to(tl.bfloat16)          # rounding step (this is the stored bf16 value)
    _xf = x_bf.to(tl.float32)          # .float()

    ms = tl.sum(_xf * _xf, axis=0) / N
    rinv = tl.math.rsqrt(ms + eps)

    y = (_xf * rinv).to(tl.bfloat16)   # cast to bf16 as in reference

    w = tl.load(W_ptr + cols, mask=mask, other=0.0)  # bf16
    yw = (y.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)  # correctly-rounded bf16 mul

    b = tl.load(B_ptr + cols, mask=mask, other=0.0)  # bf16
    out = (yw.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)  # correctly-rounded bf16 add

    tl.store(Out_ptr + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 4096, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # tensor-core bf16 GEMM
        x = x.contiguous()
        m, n = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _fused_scale_rms_bias[(m,)](
            x, self.rms2_w, self.b3, out,
            n, x.stride(0), out.stride(0),
            1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out

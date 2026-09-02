import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 568
M, D, DT = 512, 512, torch.bfloat16


@triton.jit
def _fused_gelu_rms_bias_kernel(
    X_ptr, W_ptr, B3_ptr, B4_ptr, Out_ptr,
    D, stride_row,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    # load matmul output row (bf16 -> fp32)
    x = tl.load(X_ptr + row * stride_row + cols, mask=mask, other=0.0).to(tl.float32)

    # exact (erf-based) GELU in fp32, then round to bf16 (matches PyTorch bf16 gelu)
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # RMSNorm computed in fp32 (matches _xf = x.float() path)
    ms = tl.sum(g * g, axis=0) / D
    rinv = 1.0 / tl.sqrt(ms + 1e-6)
    y = (g * rinv).to(tl.bfloat16).to(tl.float32)

    # * rms2_w (bf16 elementwise -> fp32 compute, bf16 round)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (y * w).to(tl.bfloat16).to(tl.float32)

    # + b3
    b3 = tl.load(B3_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (y + b3).to(tl.bfloat16).to(tl.float32)

    # + b4
    b4 = tl.load(B4_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (y + b4).to(tl.bfloat16)

    tl.store(Out_ptr + row * stride_row + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # Tensor-core matmul (same as reference)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        m, d = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(d)
        _fused_gelu_rms_bias_kernel[(m,)](
            h, self.rms2_w, self.b3, self.b4, out,
            d, h.stride(0),
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out

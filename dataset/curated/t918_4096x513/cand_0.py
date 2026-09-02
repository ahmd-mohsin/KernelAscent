import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 918
M, D, DT = 4096, 513, torch.float16


@triton.jit
def _fused_scale_bias_rms_gelu(
    X_ptr, B_ptr, W_ptr, O_ptr,
    N, stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # load matmul output (fp16), replicate PyTorch's per-op fp16 rounding
    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # x * 1.1963 (fp16 rounding)
    a = (x * 1.1963).to(tl.float16).to(tl.float32)

    # + b2 (fp16 rounding)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    v = (a + b).to(tl.float16).to(tl.float32)

    # RMS norm in fp32
    ms = tl.sum(v * v, axis=0) / N
    r = tl.math.rsqrt(ms + 1e-6)
    y = (v * r).to(tl.float16).to(tl.float32)

    # * rms3_w (fp16 rounding)
    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    z = (y * w).to(tl.float16).to(tl.float32)

    # exact GELU in fp32 (opmath), round to fp16
    g = 0.5 * z * (1.0 + tl.math.erf(z * 0.7071067811865476))
    g = g.to(tl.float16).to(tl.float32)

    # * 1.042, final fp16
    out = (g * 1.042).to(tl.float16)
    tl.store(O_ptr + row * stride_o + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 4096, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores)
        h = x @ self.W0  # (M, 4096) fp16
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _fused_scale_bias_rms_gelu[(m,)](
            h, self.b2, self.rms3_w, out,
            n, h.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out

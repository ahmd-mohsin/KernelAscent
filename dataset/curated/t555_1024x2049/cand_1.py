import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 555
M, D, DT = 1024, 2049, torch.float16


@triton.jit
def _rms_gelu_relu_kernel(
    X_ptr, W_ptr, Out_ptr,
    N, stride_x, stride_o,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + eps)

    # RMS normalize, cast to fp16 (match reference .to(x.dtype))
    y = (xf * inv).to(tl.float16)

    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    # elementwise mul: PyTorch computes half*half in fp32 then casts to half
    z = (y.to(tl.float32) * w.to(tl.float32)).to(tl.float16)

    # GELU (erf-based) computed in fp32, cast back to half (matches PyTorch CUDA)
    zf = z.to(tl.float32)
    g = (zf * 0.5 * (1.0 + tl.math.erf(zf * 0.7071067811865476))).to(tl.float16)

    # ReLU
    out = tl.maximum(g, tl.zeros_like(g))

    tl.store(Out_ptr + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 1024, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        m, n = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _rms_gelu_relu_kernel[(m,)](
            x, self.rms1_w, out,
            n, x.stride(0), out.stride(0),
            1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out

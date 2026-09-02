import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 635
M, D, DT = 2048, 2048, torch.bfloat16


@triton.jit
def _fused_rms_gelu_kernel(
    X, W, Y,
    stride_x, stride_y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # x = x * 1.2974  (bf16 op: fp32 math, round to bf16)
    x = (x * 1.2974).to(tl.bfloat16).to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(x * x, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    x = x * inv
    # cast to bf16 (as in reference .to(x.dtype))
    x = x.to(tl.bfloat16).to(tl.float32)

    # multiply by rms2_w (bf16 op: fp32 math, round to bf16)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    x = (x * w).to(tl.bfloat16).to(tl.float32)

    # x = x * 1.1325 (bf16 op)
    x = (x * 1.1325).to(tl.bfloat16).to(tl.float32)

    # exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    y = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    y = y.to(tl.bfloat16)

    tl.store(Y + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 GEMM
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _fused_rms_gelu_kernel[(m,)](
            h, self.rms2_w, out,
            h.stride(0), out.stride(0),
            N=n, BLOCK=BLOCK,
            num_warps=4,
        )
        return out

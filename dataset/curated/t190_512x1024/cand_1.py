import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 190
M, D, DT = 512, 1024, torch.bfloat16


@triton.jit
def _fused_scale_rms_bias_kernel(
    X_ptr, W_ptr, B_ptr, Out_ptr,
    N,
    stride_x,
    stride_o,
    EPS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # load bf16 row (matmul output)
    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0)

    # x = x * 1.0427  (computed in fp32, rounded back to bf16, as PyTorch does)
    xf = x.to(tl.float32) * SCALE
    x_bf = xf.to(tl.bfloat16)

    # RMSNorm in fp32
    xf2 = x_bf.to(tl.float32)
    ms = tl.sum(xf2 * xf2, axis=0) / N
    r = tl.math.rsqrt(ms + EPS)

    y = (xf2 * r).to(tl.bfloat16)

    w = tl.load(W_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)

    # bf16 elementwise ops with per-op rounding (matches PyTorch bf16 semantics)
    t = (y.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)
    out = (t.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)

    tl.store(Out_ptr + row * stride_o + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 4096, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul on tensor cores (identical to reference)
        y = x @ self.W0

        orig_shape = y.shape
        N = orig_shape[-1]
        y2 = y.reshape(-1, N)
        if not y2.is_contiguous():
            y2 = y2.contiguous()
        rows = y2.shape[0]

        out = torch.empty_like(y2)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_scale_rms_bias_kernel[(rows,)](
            y2, self.rms2_w, self.b3, out,
            N,
            y2.stride(0),
            out.stride(0),
            EPS=1e-6,
            SCALE=1.0427,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.reshape(orig_shape)

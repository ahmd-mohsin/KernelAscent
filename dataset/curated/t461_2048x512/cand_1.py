import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 461
M, D, DT = 2048, 512, torch.bfloat16


@triton.jit
def _scale_rms_bias_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    stride_x, stride_y,
    N, eps, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # x = x * 1.0961  (computed on bf16 tensor -> round to bf16, matching PyTorch)
    xs = (x * scale).to(tl.bfloat16).to(tl.float32)

    # RMS norm in fp32
    ms = tl.sum(xs * xs, axis=0) / N
    rs = tl.rsqrt(ms + eps)

    # (_xf * rsqrt).to(bf16)
    y = (xs * rs).to(tl.bfloat16).to(tl.float32)

    # * rms2_w  (bf16 elementwise: fp32 compute, bf16 round)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (y * w).to(tl.bfloat16).to(tl.float32)

    # + b3
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (y + b).to(tl.bfloat16)

    tl.store(Y_ptr + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS matmul (already optimal on A100 tensor cores)
        h = x @ self.W0

        orig_shape = h.shape
        h2 = h.reshape(-1, orig_shape[-1])
        if not h2.is_contiguous():
            h2 = h2.contiguous()

        rows, N = h2.shape
        out = torch.empty_like(h2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 4096 else 4

        _scale_rms_bias_kernel[(rows,)](
            h2, self.rms2_w, self.b3, out,
            h2.stride(0), out.stride(0),
            N, 1e-6, 1.0961,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.reshape(orig_shape)

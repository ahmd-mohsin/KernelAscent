import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 417
M, D, DT = 8192, 513, torch.bfloat16


@triton.jit
def _fused_bias_rms_kernel(
    x_ptr, b0_ptr, b1_ptr, b2_ptr, w_ptr, out_ptr,
    n_cols, x_stride, out_stride,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * x_stride + cols, mask=mask, other=0.0)
    b0 = tl.load(b0_ptr + cols, mask=mask, other=0.0)
    b1 = tl.load(b1_ptr + cols, mask=mask, other=0.0)
    b2 = tl.load(b2_ptr + cols, mask=mask, other=0.0)

    # Emulate sequential bf16 additions (round to bf16 after each add)
    x = (x.to(tl.float32) + b0.to(tl.float32)).to(tl.bfloat16)
    x = (x.to(tl.float32) + b1.to(tl.float32)).to(tl.bfloat16)
    x = (x.to(tl.float32) + b2.to(tl.float32)).to(tl.bfloat16)

    xf = x.to(tl.float32)
    xf = tl.where(mask, xf, 0.0)

    mean_sq = tl.sum(xf * xf, axis=0) / n_cols
    r = 1.0 / tl.sqrt(mean_sq + eps)

    y = (xf * r).to(tl.bfloat16)  # round to bf16 like reference .to(x.dtype)

    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    out = (y.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    tl.store(out_ptr + row * out_stride + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        n_cols = orig_shape[-1]
        x2d = x.contiguous().view(-1, n_cols)
        n_rows = x2d.shape[0]
        out = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 4 if BLOCK <= 1024 else 8

        _fused_bias_rms_kernel[(n_rows,)](
            x2d, self.b0, self.b1, self.b2, self.rms3_w, out,
            n_cols, x2d.stride(0), out.stride(0),
            1e-6,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)

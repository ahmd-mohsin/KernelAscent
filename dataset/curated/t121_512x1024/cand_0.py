import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 121
M, D, DT = 512, 1024, torch.bfloat16


@triton.jit
def _fused_relu_rms_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    D: tl.constexpr,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(x_ptr + row * D + offs, mask=mask, other=0.0)
    # relu in bf16 (identical result), then float
    xf = tl.maximum(x, 0.0).to(tl.float32)

    mean_sq = tl.sum(xf * xf, axis=0) / D
    inv = 1.0 / tl.sqrt(mean_sq + eps)

    # normalize in fp32, round to bf16 (matches .to(x.dtype))
    xn = (xf * inv).to(tl.bfloat16)

    w = tl.load(w_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)

    # bf16 * bf16 -> compute in fp32, round back to bf16 (PyTorch semantics)
    y = (xn.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)
    y = (y.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)

    tl.store(out_ptr + row * D + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        Mrows, Dcols = x.shape[-2] if x.dim() > 1 else 1, x.shape[-1]
        x2d = x.view(-1, Dcols)
        n_rows = x2d.shape[0]
        out = torch.empty_like(x2d)
        BLOCK = triton.next_power_of_2(Dcols)
        _fused_relu_rms_kernel[(n_rows,)](
            x2d, self.rms1_w, self.b2, out,
            D=Dcols, eps=1e-6, BLOCK=BLOCK,
            num_warps=8,
        )
        return out.view(x.shape)

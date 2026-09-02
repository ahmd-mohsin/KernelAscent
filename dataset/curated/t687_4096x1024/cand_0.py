import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 687
M, D, DT = 4096, 1024, torch.bfloat16


@triton.jit
def _gelu_rms_kernel(
    x_ptr, w_ptr, out_ptr,
    D: tl.constexpr,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(x_ptr + row * D + cols, mask=mask, other=0.0).to(tl.float32)

    # exact GELU in fp32 (matches F.gelu which upcasts bf16 to float internally)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))

    # round to bf16 (F.gelu output dtype), then upcast to fp32 like _xf = x.float()
    g_bf = g.to(tl.bfloat16)
    xf = g_bf.to(tl.float32)

    ms = tl.sum(tl.where(mask, xf * xf, 0.0), axis=0) / D
    inv = tl.math.rsqrt(ms + eps)

    normed = (xf * inv).to(tl.bfloat16)

    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    # bf16 * bf16 in PyTorch CUDA computes in fp32 (opmath) then casts back
    out = (normed.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    tl.store(out_ptr + row * D + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(d)
        _gelu_rms_kernel[(m,)](
            x2, self.rms1_w, out,
            D=d, eps=1e-6, BLOCK=BLOCK,
            num_warps=8,
        )
        return out.view(orig_shape)

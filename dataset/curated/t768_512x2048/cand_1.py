import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 768
M, D, DT = 512, 2048, torch.float16


@triton.jit
def _fused_row_kernel(
    x_ptr, w_ptr, out_ptr,
    n_cols,
    stride_row,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    ptr = x_ptr + row * stride_row + cols
    x = tl.load(ptr, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax (fp32 accumulation, like PyTorch fp16 softmax)
    row_max = tl.max(x, axis=0)
    ex = tl.exp(x - row_max)
    ex = tl.where(mask, ex, 0.0)
    denom = tl.sum(ex, axis=0)
    s = ex / denom
    s = s.to(tl.float16)  # round to fp16 (intermediate storage)

    # gelu (exact, erf-based) in fp32 opmath
    sf = s.to(tl.float32)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = 0.5 * sf * (1.0 + tl.math.erf(sf * INV_SQRT2))
    g = g.to(tl.float16)

    # scale
    y = (g.to(tl.float32) * 1.1037).to(tl.float16)

    # RMSNorm in fp32
    yf = y.to(tl.float32)
    yf = tl.where(mask, yf, 0.0)
    ms = tl.sum(yf * yf, axis=0) / n_cols
    inv = tl.math.rsqrt(ms + 1e-6)
    normed = (yf * inv).to(tl.float16)

    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float16)
    z = (normed.to(tl.float32) * w.to(tl.float32)).to(tl.float16)

    # relu
    z = tl.maximum(z, 0.0)

    tl.store(out_ptr + row * stride_row + cols, z, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms3_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda or x.dtype != torch.float16:
            # fallback path (reference implementation)
            x = torch.softmax(x, dim=-1)
            x = F.gelu(x)
            x = x * 1.1037
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms3_w
            return torch.relu(x)

        x = x.contiguous()
        orig_shape = x.shape
        n_cols = orig_shape[-1]
        x2 = x.view(-1, n_cols)
        n_rows = x2.shape[0]
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_row_kernel[(n_rows,)](
            x2, self.rms3_w, out,
            n_cols, x2.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.view(orig_shape)

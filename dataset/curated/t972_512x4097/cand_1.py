import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 972
M, D, DT = 512, 4097, torch.bfloat16


@triton.jit
def _fused_gelu_softmax_ln_kernel(
    X_ptr, OUT_ptr, G_ptr, B_ptr,
    n_cols, stride_row,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols

    x = tl.load(X_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)

    # exact GELU (erf-based), computed in fp32, rounded back to bf16 (matches PyTorch bf16 gelu)
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # scale by 1.2255 (opmath fp32, result rounded to bf16 as PyTorch does)
    y = (g * 1.2255).to(tl.bfloat16).to(tl.float32)

    # softmax over the row (fp32 accumulation, bf16 output rounding)
    y_masked = tl.where(mask, y, float('-inf'))
    row_max = tl.max(y_masked, axis=0)
    e = tl.exp(y_masked - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    sm = (e / denom).to(tl.bfloat16).to(tl.float32)

    # layer norm (fp32 statistics, matches PyTorch's accumulation type)
    n = n_cols.to(tl.float32)
    mean = tl.sum(tl.where(mask, sm, 0.0), axis=0) / n
    diff = tl.where(mask, sm - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / n
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    out = diff * rstd * w + b
    tl.store(OUT_ptr + row * stride_row + offs, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln3_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = F.gelu(x)
            x = x * 1.2255
            x = torch.softmax(x, dim=-1)
            x = F.layer_norm(x, (x.shape[-1],), self.ln3_g, self.ln3_b)
            return x

        orig_shape = x.shape
        n_cols = orig_shape[-1]
        x2d = x.contiguous().view(-1, n_cols)
        n_rows = x2d.shape[0]
        out = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 8 if BLOCK <= 4096 else 16

        _fused_gelu_softmax_ln_kernel[(n_rows,)](
            x2d, out,
            self.ln3_g, self.ln3_b,
            n_cols, x2d.stride(0),
            1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 444
M, D, DT = 2048, 2049, torch.bfloat16


@triton.jit
def _fused_ln_softmax_kernel(
    x_ptr, g_ptr, b_ptr, b2_ptr, out_ptr,
    N, x_stride, out_stride, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(x_ptr + row * x_stride + offs, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm (fp32 accumulation, matching PyTorch) ----
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (x - mean) * rstd * g + b
    # match PyTorch's bf16 intermediate rounding after layer_norm
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- Softmax (fp32 accumulation, matching PyTorch) ----
    y_masked = tl.where(mask, y, float('-inf'))
    row_max = tl.max(y_masked, axis=0)
    e = tl.exp(y_masked - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    sm = e / denom
    # match PyTorch's bf16 intermediate rounding after softmax
    sm = sm.to(tl.bfloat16).to(tl.float32)

    # ---- add bias, scale (each op rounds to bf16, matching PyTorch) ----
    b2 = tl.load(b2_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    z = (sm + b2).to(tl.bfloat16).to(tl.float32)
    out = (z * 1.3125).to(tl.bfloat16)

    tl.store(out_ptr + row * out_stride + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
            y = torch.softmax(y, dim=-1)
            y = y + self.b2
            y = y * 1.3125
            return y

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_ln_softmax_kernel[(rows,)](
            x2, self.ln0_g, self.ln0_b, self.b2, out,
            N, x2.stride(0), out.stride(0), 1e-5,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.view(orig_shape)

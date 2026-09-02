import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 513
M, D, DT = 512, 1025, torch.float16


@triton.jit
def _fused_kernel(
    x_ptr, g_ptr, b_ptr, b3_ptr, out_ptr,
    D, stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # --- GELU (erf-based, computed in fp32 like PyTorch opmath, output fp16) ---
    x = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    x = x.to(tl.float16).to(tl.float32)

    # --- Softmax (fp32 accumulation, fp16 output) ---
    xm = tl.where(mask, x, float('-inf'))
    mx = tl.max(xm, 0)
    e = tl.exp(xm - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, 0)
    x = e / s
    x = x.to(tl.float16).to(tl.float32)

    # --- LayerNorm (fp32 stats, fp16 output) ---
    mean = tl.sum(tl.where(mask, x, 0.0), 0) / D
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, 0) / D
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = diff * rstd * g + b
    y = y.to(tl.float16).to(tl.float32)

    # --- Add bias (fp16 rounding) ---
    b3 = tl.load(b3_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = y + b3
    y = y.to(tl.float16).to(tl.float32)

    # --- Softmax (fp32 accumulation, fp16 output) ---
    ym = tl.where(mask, y, float('-inf'))
    my = tl.max(ym, 0)
    e2 = tl.exp(ym - my)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, 0)
    o = e2 / s2

    tl.store(out_ptr + row * stride_o + offs, o.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln2_g = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = F.gelu(x)
            x = torch.softmax(x, dim=-1)
            x = F.layer_norm(x, (x.shape[-1],), self.ln2_g, self.ln2_b)
            x = x + self.b3
            return torch.softmax(x, dim=-1)

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        _fused_kernel[(m,)](
            x2, self.ln2_g, self.ln2_b, self.b3, out,
            d, x2.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out.view(orig_shape)

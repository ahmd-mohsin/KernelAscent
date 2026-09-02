import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 418
M, D, DT = 2048, 2049, torch.float16


@triton.jit
def _fused_kernel(
    x_ptr, b0_ptr, g_ptr, b_ptr, out_ptr,
    N, stride_x, stride_o,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # --- add bias in fp16 (matches reference: x = x + b0 in half) ---
    x = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=0.0)
    b0 = tl.load(b0_ptr + offs, mask=mask, other=0.0)
    x = x + b0  # fp16 add

    xf = x.to(tl.float32)

    # --- layer norm (fp32 accumulation, cast to fp16 like PyTorch) ---
    xf_masked = tl.where(mask, xf, 0.0)
    mean = tl.sum(xf_masked, axis=0) / N
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = diff * rstd * g + b
    y = y.to(tl.float16).to(tl.float32)  # round to fp16 as reference does

    # --- softmax #1 (fp32 accumulation, output rounded to fp16) ---
    y1 = tl.where(mask, y, float("-inf"))
    m1 = tl.max(y1, axis=0)
    e1 = tl.exp(y1 - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, axis=0)
    y = (e1 / s1).to(tl.float16).to(tl.float32)

    # --- softmax #2 ---
    y2 = tl.where(mask, y, float("-inf"))
    m2 = tl.max(y2, axis=0)
    e2 = tl.exp(y2 - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    out = (e2 / s2).to(tl.float16)

    tl.store(out_ptr + row * stride_o + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x + self.b0
            x = F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)
            x = torch.softmax(x, dim=-1)
            x = torch.softmax(x, dim=-1)
            return x

        orig_shape = x.shape
        N = orig_shape[-1]
        x2d = x.contiguous().view(-1, N)
        rows = x2d.shape[0]
        out = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_kernel[(rows,)](
            x2d, self.b0, self.ln1_g, self.ln1_b, out,
            N, x2d.stride(0), out.stride(0),
            EPS=1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)

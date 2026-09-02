import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 306
M, D, DT = 4096, 1025, torch.float16


@triton.jit
def _fused_ln_softmax_relu_softmax(
    x_ptr, g_ptr, b_ptr, b2_ptr, out_ptr,
    n_cols, x_stride, out_stride, EPS,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols

    # ---- LayerNorm (fp32 accumulation, like PyTorch half layer_norm) ----
    x = tl.load(x_ptr + row * x_stride + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / n_cols
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + EPS)

    g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = diff * rstd * g + b
    # round to fp16 as PyTorch materializes fp16 between ops
    y = y.to(tl.float16).to(tl.float32)

    # ---- Softmax 1 (fp32 math on fp16-rounded input) ----
    y = tl.where(mask, y, float('-inf'))
    m1 = tl.max(y, axis=0)
    e1 = tl.exp(y - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, axis=0)
    y16 = (e1 / s1).to(tl.float16)

    # ---- Add b2 + ReLU in fp16 (matches reference fp16 arithmetic) ----
    b2 = tl.load(b2_ptr + offs, mask=mask, other=0.0)  # fp16
    y16 = y16 + b2
    zero16 = tl.zeros([BLOCK], dtype=tl.float16)
    y16 = tl.maximum(y16, zero16)

    # ---- Softmax 2 (fp32 math on fp16 input) ----
    y2 = y16.to(tl.float32)
    y2 = tl.where(mask, y2, float('-inf'))
    m2 = tl.max(y2, axis=0)
    e2 = tl.exp(y2 - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    out = (e2 / s2).to(tl.float16)

    tl.store(out_ptr + row * out_stride + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
            x = torch.softmax(x, dim=-1)
            x = x + self.b2
            x = torch.relu(x)
            x = torch.softmax(x, dim=-1)
            return x

        orig_shape = x.shape
        n_cols = orig_shape[-1]
        x2d = x.contiguous().view(-1, n_cols)
        n_rows = x2d.shape[0]
        out = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16

        _fused_ln_softmax_relu_softmax[(n_rows,)](
            x2d, self.ln0_g, self.ln0_b, self.b2, out,
            n_cols, x2d.stride(0), out.stride(0), 1e-5,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.view(orig_shape)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 196
M, D, DT = 8192, 512, torch.float16


@triton.jit
def fused_kernel(x_ptr, b0_ptr, g_ptr, b_ptr, b3_ptr, out_ptr,
                 N, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    base = row * D

    x = tl.load(x_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(b0_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # x = x + b0 (rounded to fp16 as in reference)
    x = (x + b0).to(tl.float16).to(tl.float32)

    # LayerNorm (fp32 accumulation, fp16 output)
    mean = tl.sum(x, axis=0) / D
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(g_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x = ((x - mean) * rstd * g + b).to(tl.float16).to(tl.float32)

    # softmax 1 (fp32 accumulation, fp16 output)
    m1 = tl.max(tl.where(mask, x, float('-inf')), axis=0)
    e1 = tl.where(mask, tl.exp(x - m1), 0.0)
    s1 = tl.sum(e1, axis=0)
    x = (e1 / s1).to(tl.float16).to(tl.float32)

    # + b3 (fp16 rounding)
    b3 = tl.load(b3_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x = (x + b3).to(tl.float16).to(tl.float32)

    # softmax 2
    m2 = tl.max(tl.where(mask, x, float('-inf')), axis=0)
    e2 = tl.where(mask, tl.exp(x - m2), 0.0)
    s2 = tl.sum(e2, axis=0)
    y = (e2 / s2).to(tl.float16)

    tl.store(out_ptr + base + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = x + self.b0
            y = F.layer_norm(y, (y.shape[-1],), self.ln1_g, self.ln1_b)
            y = torch.softmax(y, dim=-1)
            y = y + self.b3
            return torch.softmax(y, dim=-1)

        x = x.contiguous()
        orig_shape = x.shape
        Dd = orig_shape[-1]
        x2 = x.view(-1, Dd)
        n_rows = x2.shape[0]
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(Dd)
        fused_kernel[(n_rows,)](
            x2, self.b0, self.ln1_g, self.ln1_b, self.b3, out,
            n_rows, D=Dd, BLOCK=BLOCK,
            num_warps=8,
        )
        return out.view(orig_shape)

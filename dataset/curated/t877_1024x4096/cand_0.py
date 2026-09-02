import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 877
M, D, DT = 1024, 4096, torch.bfloat16


@triton.jit
def _fused_kernel(
    x_ptr, out_ptr,
    g1_ptr, b1_ptr, g2_ptr, b2_ptr,
    N, stride_x, stride_o,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=-float('inf')).to(tl.float32)

    # ---- softmax (fp32 accumulation, round to bf16 like PyTorch output) ----
    mx = tl.max(x, axis=0)
    e = tl.exp(x - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- layernorm 1 ----
    n_f = N.to(tl.float32)
    mean1 = tl.sum(tl.where(mask, y, 0.0), axis=0) / n_f
    d1 = tl.where(mask, y - mean1, 0.0)
    var1 = tl.sum(d1 * d1, axis=0) / n_f
    rstd1 = 1.0 / tl.sqrt(var1 + eps)
    g1 = tl.load(g1_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(b1_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = d1 * rstd1 * g1 + b1
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- layernorm 2 ----
    mean2 = tl.sum(tl.where(mask, y, 0.0), axis=0) / n_f
    d2 = tl.where(mask, y - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / n_f
    rstd2 = 1.0 / tl.sqrt(var2 + eps)
    g2 = tl.load(g2_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(b2_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = d2 * rstd2 * g2 + b2
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- relu ----
    y = tl.maximum(y, 0.0)

    tl.store(out_ptr + row * stride_o + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = torch.softmax(x, dim=-1)
            y = F.layer_norm(y, (y.shape[-1],), self.ln1_g, self.ln1_b)
            y = F.layer_norm(y, (y.shape[-1],), self.ln2_g, self.ln2_b)
            return torch.relu(y)

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_kernel[(rows,)](
            x2, out,
            self.ln1_g, self.ln1_b, self.ln2_g, self.ln2_b,
            N, x2.stride(0), out.stride(0),
            1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)

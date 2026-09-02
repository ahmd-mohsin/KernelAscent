import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 432
M, D, DT = 8192, 4097, torch.bfloat16


@triton.jit
def _fused_row_kernel(
    X_ptr, B2_ptr, G3_ptr, B3_ptr, G4_ptr, B4_ptr, Y_ptr,
    N, stride_x, stride_y,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # relu
    x = tl.maximum(x, 0.0)
    # softmax (masked lanes -> -inf so exp = 0)
    x = tl.where(mask, x, float('-inf'))
    row_max = tl.max(x, axis=0)
    num = tl.exp(x - row_max)
    denom = tl.sum(num, axis=0)
    p = num / denom
    # round to bf16 (matches PyTorch's softmax output dtype)
    p = p.to(tl.bfloat16).to(tl.float32)

    # + b2 (bf16 rounding after add, like PyTorch bf16 elementwise add)
    b2 = tl.load(B2_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x = p + b2
    x = x.to(tl.bfloat16).to(tl.float32)

    # LayerNorm 1 (fp32 stats, bf16 output)
    n_f = N.to(tl.float32)
    xm = tl.where(mask, x, 0.0)
    mean1 = tl.sum(xm, axis=0) / n_f
    d1 = tl.where(mask, x - mean1, 0.0)
    var1 = tl.sum(d1 * d1, axis=0) / n_f
    rstd1 = 1.0 / tl.sqrt(var1 + EPS)
    g3 = tl.load(G3_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x = d1 * rstd1 * g3 + b3
    x = x.to(tl.bfloat16).to(tl.float32)

    # LayerNorm 2 (fp32 stats, bf16 output)
    xm = tl.where(mask, x, 0.0)
    mean2 = tl.sum(xm, axis=0) / n_f
    d2 = tl.where(mask, x - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / n_f
    rstd2 = 1.0 / tl.sqrt(var2 + EPS)
    g4 = tl.load(G4_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(B4_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = d2 * rstd2 * g4 + b4

    tl.store(Y_ptr + row * stride_y + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b2 = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = torch.relu(x)
            x = torch.softmax(x, dim=-1)
            x = x + self.b2
            x = F.layer_norm(x, (x.shape[-1],), self.ln3_g, self.ln3_b)
            x = F.layer_norm(x, (x.shape[-1],), self.ln4_g, self.ln4_b)
            return x

        orig_shape = x.shape
        N = orig_shape[-1]
        x2d = x.contiguous().view(-1, N)
        rows = x2d.shape[0]
        y = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 16 if BLOCK >= 8192 else 8

        _fused_row_kernel[(rows,)](
            x2d, self.b2, self.ln3_g, self.ln3_b, self.ln4_g, self.ln4_b, y,
            N, x2d.stride(0), y.stride(0),
            EPS=1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 432
M, D, DT = 8192, 4097, torch.bfloat16


@triton.jit
def _fused_kernel(
    X_ptr, B2_ptr, G3_ptr, B3_ptr, G4_ptr, B4_ptr, Out_ptr,
    N,  # row length (4097)
    stride_x, stride_o,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x_ptrs = X_ptr + row * stride_x + cols
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    # ReLU
    x = tl.maximum(x, 0.0)

    # Softmax (fp32 accumulation, bf16 output like PyTorch)
    xm = tl.where(mask, x, float('-inf'))
    row_max = tl.max(xm, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    sm = e / denom
    # round to bf16 as PyTorch would output bf16 tensor
    sm = sm.to(tl.bfloat16).to(tl.float32)

    # + b2 (bf16 add -> bf16 result)
    b2 = tl.load(B2_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (sm + b2)
    y = y.to(tl.bfloat16).to(tl.float32)
    y = tl.where(mask, y, 0.0)

    n_f = N.to(tl.float32)

    # LayerNorm 3 (fp32 stats, bf16 output)
    mean1 = tl.sum(y, axis=0) / n_f
    d1 = tl.where(mask, y - mean1, 0.0)
    var1 = tl.sum(d1 * d1, axis=0) / n_f
    rstd1 = 1.0 / tl.sqrt(var1 + EPS)
    g3 = tl.load(G3_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    z = d1 * rstd1 * g3 + b3
    z = z.to(tl.bfloat16).to(tl.float32)
    z = tl.where(mask, z, 0.0)

    # LayerNorm 4
    mean2 = tl.sum(z, axis=0) / n_f
    d2 = tl.where(mask, z - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / n_f
    rstd2 = 1.0 / tl.sqrt(var2 + EPS)
    g4 = tl.load(G4_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(B4_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = d2 * rstd2 * g4 + b4

    tl.store(Out_ptr + row * stride_o + cols, out.to(tl.bfloat16), mask=mask)


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
            y = torch.relu(x)
            y = torch.softmax(y, dim=-1)
            y = y + self.b2
            y = F.layer_norm(y, (y.shape[-1],), self.ln3_g, self.ln3_b)
            y = F.layer_norm(y, (y.shape[-1],), self.ln4_g, self.ln4_b)
            return y

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 4096 else 4

        _fused_kernel[(rows,)](
            x2, self.b2, self.ln3_g, self.ln3_b, self.ln4_g, self.ln4_b, out,
            N,
            x2.stride(0), out.stride(0),
            EPS=1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)

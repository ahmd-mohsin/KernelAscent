import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 556
M, D, DT = 4096, 1024, torch.bfloat16


@triton.jit
def _softmax_ln_bias_kernel(
    X_ptr, G_ptr, B_ptr, B4_ptr, Y_ptr,
    stride_x, stride_y,
    N: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax (fp32 accumulate, like PyTorch)
    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    p = e / denom

    # relu is a no-op on softmax output (all values >= 0)

    # cast to bf16 (softmax output dtype) then back to fp32 for layernorm stats
    p = p.to(tl.bfloat16).to(tl.float32)

    mean = tl.sum(p, axis=0) / N
    d = tl.where(mask, p - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    g = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g + b
    y_bf = y.to(tl.bfloat16)

    b4 = tl.load(B4_ptr + offs, mask=mask, other=0.0)
    out = y_bf + b4

    tl.store(Y_ptr + row * stride_y + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x @ self.W0
            x = torch.softmax(x, dim=-1)
            x = torch.relu(x)
            x = F.layer_norm(x, (x.shape[-1],), self.ln3_g, self.ln3_b)
            return x + self.b4

        h = torch.matmul(x, self.W0)  # cuBLAS bf16 GEMM
        if not h.is_contiguous():
            h = h.contiguous()

        Mrows, N = h.shape
        y = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        _softmax_ln_bias_kernel[(Mrows,)](
            h, self.ln3_g, self.ln3_b, self.b4, y,
            h.stride(0), y.stride(0),
            N=N, EPS=1e-5, BLOCK=BLOCK,
            num_warps=4,
        )
        return y

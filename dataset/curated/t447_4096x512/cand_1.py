import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 447
M, D, DT = 4096, 512, torch.bfloat16


@triton.jit
def _softmax_kernel(X, Y, N, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(Y + row * stride_y + cols, y.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _layernorm_kernel(X, Y, G, B, N, stride_x, stride_y, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g + b
    tl.store(Y + row * stride_y + cols, y.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.W2 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            h = x @ self.W0
            h = torch.softmax(h, dim=-1)
            h = h @ self.W2
            return F.layer_norm(h, (h.shape[-1],), self.ln3_g, self.ln3_b)

        # GEMM 1 (cuBLAS tensor cores)
        h = x @ self.W0  # (M, 4096)
        rows, n1 = h.shape

        # Fused softmax (single-pass Triton kernel, fp32 accumulation)
        _softmax_kernel[(rows,)](
            h, h, n1, h.stride(0), h.stride(0),
            BLOCK=triton.next_power_of_2(n1), num_warps=8,
        )

        # GEMM 2 (cuBLAS tensor cores)
        o = h @ self.W2  # (M, 1024)
        n2 = o.shape[1]
        y = torch.empty_like(o)

        # Fused layernorm (single-pass Triton kernel, fp32 accumulation)
        _layernorm_kernel[(rows,)](
            o, y, self.ln3_g, self.ln3_b, n2, o.stride(0), y.stride(0),
            1e-5, BLOCK=triton.next_power_of_2(n2), num_warps=4,
        )
        return y

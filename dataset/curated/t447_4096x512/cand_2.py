import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 447
M, D, DT = 4096, 512, torch.bfloat16


@triton.jit
def _softmax_kernel(X, Y, N, stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + row * stride + offs, mask=mask, other=float('-inf')).to(tl.float32)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(Y + row * stride + offs, y.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _layernorm_kernel(X, Y, G, B, N, stride, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + row * stride + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g + b
    tl.store(Y + row * stride + offs, y.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.W2 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM 1 (cuBLAS tensor cores)
        h = torch.matmul(x, self.W0)  # (M, 4096)
        h = h.contiguous()
        Mrows, N1 = h.shape

        if h.is_cuda:
            # Fused softmax (in-place, single pass, fp32 accumulation)
            BLOCK1 = triton.next_power_of_2(N1)
            _softmax_kernel[(Mrows,)](
                h, h, N1, h.stride(0),
                BLOCK=BLOCK1, num_warps=8,
            )

            # GEMM 2 (cuBLAS tensor cores)
            out = torch.matmul(h, self.W2)  # (M, 1024)
            out = out.contiguous()
            N2 = out.shape[-1]

            # Fused layernorm (in-place, fp32 accumulation)
            BLOCK2 = triton.next_power_of_2(N2)
            _layernorm_kernel[(Mrows,)](
                out, out, self.ln3_g, self.ln3_b,
                N2, out.stride(0), 1e-5,
                BLOCK=BLOCK2, num_warps=4,
            )
            return out
        else:
            h = torch.softmax(h, dim=-1)
            out = torch.matmul(h, self.W2)
            return F.layer_norm(out, (out.shape[-1],), self.ln3_g, self.ln3_b)

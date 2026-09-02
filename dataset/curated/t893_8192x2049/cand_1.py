import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 893
M, D, DT = 8192, 2049, torch.bfloat16


@triton.jit
def _ln_kernel(X, G, B, Y, D, stride_x, stride_y, eps,
               BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / D
    xm = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xm * xm, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = xm * rstd * g + b
    tl.store(Y + row * stride_y + cols, y.to(tl.bfloat16), mask=mask)


@triton.jit
def _ln_relu_softmax_kernel(X, G, B, Y, D, stride_x, stride_y, eps,
                            BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    x = tl.load(X + row * stride_x + cols).to(tl.float32)
    mean = tl.sum(x, axis=0) / D
    xm = x - mean
    var = tl.sum(xm * xm, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(G + cols).to(tl.float32)
    b = tl.load(B + cols).to(tl.float32)
    # match reference: layernorm output is rounded to bf16 before relu/softmax
    y = (xm * rstd * g + b).to(tl.bfloat16).to(tl.float32)
    y = tl.maximum(y, 0.0)
    m = tl.max(y, axis=0)
    e = tl.exp(y - m)
    s = tl.sum(e, axis=0)
    out = e / s
    tl.store(Y + row * stride_y + cols, out.to(tl.bfloat16))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.W1 = nn.Parameter((torch.randn(2049, 1024, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]

        if not x2.is_cuda:
            # CPU fallback (reference path)
            h = F.layer_norm(x2, (d,), self.ln0_g, self.ln0_b)
            z = h @ self.W1
            z = F.layer_norm(z, (z.shape[-1],), self.ln2_g, self.ln2_b)
            z = torch.relu(z)
            z = torch.softmax(z, dim=-1)
            return z.view(*orig_shape[:-1], z.shape[-1])

        # Fused LayerNorm (D=2049)
        h = torch.empty_like(x2)
        BLOCK0 = triton.next_power_of_2(d)
        _ln_kernel[(m,)](
            x2, self.ln0_g, self.ln0_b, h,
            d, x2.stride(0), h.stride(0), 1e-5,
            BLOCK=BLOCK0, num_warps=16,
        )

        # GEMM via cuBLAS tensor cores
        z = h @ self.W1
        n = z.shape[-1]

        # Fused LayerNorm + ReLU + Softmax (N=1024)
        out = torch.empty_like(z)
        BLOCK1 = triton.next_power_of_2(n)
        _ln_relu_softmax_kernel[(m,)](
            z, self.ln2_g, self.ln2_b, out,
            n, z.stride(0), out.stride(0), 1e-5,
            BLOCK=BLOCK1, num_warps=8,
        )

        return out.view(*orig_shape[:-1], n)

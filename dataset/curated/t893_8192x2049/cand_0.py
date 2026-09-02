import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 893
M, D, DT = 8192, 2049, torch.bfloat16


@triton.jit
def _ln_kernel(X, Y, G, B, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, 0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, 0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g + b
    tl.store(Y + row * N + cols, y.to(tl.bfloat16), mask=mask)


@triton.jit
def _ln_relu_softmax_kernel(X, Y, G, B, N: tl.constexpr, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    x = tl.load(X + row * N + cols).to(tl.float32)
    mean = tl.sum(x, 0) / N
    d = x - mean
    var = tl.sum(d * d, 0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(G + cols).to(tl.float32)
    b = tl.load(B + cols).to(tl.float32)
    # round LN output to bf16 first (matches reference dtype behavior), then relu
    y = (d * rstd * g + b).to(tl.bfloat16)
    yf = y.to(tl.float32)
    yf = tl.maximum(yf, 0.0)
    # softmax in fp32 (matches PyTorch's internal upcast for bf16 softmax)
    m = tl.max(yf, 0)
    e = tl.exp(yf - m)
    s = tl.sum(e, 0)
    out = e / s
    tl.store(Y + row * N + cols, out.to(tl.bfloat16))


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
        D_ = orig_shape[-1]
        x2d = x.contiguous().view(-1, D_)
        rows = x2d.shape[0]

        # LayerNorm 0 (Triton)
        x1 = torch.empty_like(x2d)
        BLOCK0 = triton.next_power_of_2(D_)
        _ln_kernel[(rows,)](
            x2d, x1, self.ln0_g, self.ln0_b, D_, 1e-5,
            BLOCK=BLOCK0, num_warps=8,
        )

        # Matmul (cuBLAS tensor cores)
        h = x1 @ self.W1
        h = h.contiguous()
        N2 = h.shape[-1]

        # Fused LayerNorm + ReLU + Softmax (Triton)
        out = torch.empty_like(h)
        _ln_relu_softmax_kernel[(rows,)](
            h, out, self.ln2_g, self.ln2_b, N2, 1e-5,
            BLOCK=N2, num_warps=4,
        )

        return out.view(*orig_shape[:-1], N2)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 131
M, D, DT = 2048, 2048, torch.float16


@triton.jit
def _bias_softmax_ln_kernel(X, B, G, Beta, Y, N, eps,
                            BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + row * N + offs, mask=mask, other=float('-inf')).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    x = x + b
    # softmax (fp32 accumulation, matching PyTorch fp16 softmax internals)
    m = tl.max(x, 0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, 0)
    sm = e / s
    # cast to fp16 to match intermediate storage in the reference
    sm = sm.to(tl.float16).to(tl.float32)
    # layernorm (biased variance, eps inside sqrt) in fp32
    mean = tl.sum(sm, 0) / N
    d = tl.where(mask, sm - mean, 0.0)
    var = tl.sum(d * d, 0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(Beta + offs, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g + beta
    tl.store(Y + row * N + offs, y.to(tl.float16), mask=mask)


@triton.jit
def _ln_kernel(X, G, Beta, Y, N, eps,
               BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, 0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, 0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(Beta + offs, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g + beta
    tl.store(Y + row * N + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.W4 = nn.Parameter((torch.randn(4096, 4096, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.ln5_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln5_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        x2d = x.reshape(-1, orig_shape[-1])

        # GEMM 1 (cuBLAS tensor cores)
        h = torch.matmul(x2d, self.W0)
        h = h.contiguous()

        rows, N = h.shape
        BLOCK = triton.next_power_of_2(N)
        eps = 1e-5

        # Fused: bias add + softmax + layernorm
        t = torch.empty_like(h)
        _bias_softmax_ln_kernel[(rows,)](
            h, self.b1, self.ln3_g, self.ln3_b, t, N, eps,
            BLOCK=BLOCK, num_warps=16,
        )

        # GEMM 2
        out = torch.matmul(t, self.W4)
        out = out.contiguous()

        # Final layernorm
        y = torch.empty_like(out)
        _ln_kernel[(rows,)](
            out, self.ln5_g, self.ln5_b, y, out.shape[1], eps,
            BLOCK=triton.next_power_of_2(out.shape[1]), num_warps=16,
        )

        return y.reshape(*orig_shape[:-1], y.shape[-1])

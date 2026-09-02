import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 857
M, D, DT = 4096, 4096, torch.float16


@triton.jit
def _fused_ln_softmax3_kernel(
    X_ptr, G_ptr, B_ptr, Y_ptr,
    N, eps, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    ptr = X_ptr + row * N + offs

    x = tl.load(ptr).to(tl.float32)

    # LayerNorm (fp32 accumulate, round to fp16 like PyTorch)
    mean = tl.sum(x, axis=0) / N
    d = x - mean
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(G_ptr + offs).to(tl.float32)
    b = tl.load(B_ptr + offs).to(tl.float32)
    y = d * rstd * g + b
    y = y.to(tl.float16).to(tl.float32)

    # Softmax 1 (fp32 compute, fp16 output like PyTorch)
    y = y - tl.max(y, axis=0)
    e = tl.exp(y)
    y = e / tl.sum(e, axis=0)
    y = y.to(tl.float16).to(tl.float32)

    # Softmax 2
    y = y - tl.max(y, axis=0)
    e = tl.exp(y)
    y = e / tl.sum(e, axis=0)
    y = y.to(tl.float16).to(tl.float32)

    # Scale (rounds to fp16 like PyTorch elementwise mul)
    y = (y * scale).to(tl.float16).to(tl.float32)

    # Softmax 3
    y = y - tl.max(y, axis=0)
    e = tl.exp(y)
    y = e / tl.sum(e, axis=0)

    tl.store(Y_ptr + row * N + offs, y.to(tl.float16))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS GEMM (tensor cores on A100)
        h = x @ self.W0
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        _fused_ln_softmax3_kernel[(m,)](
            h, self.ln1_g, self.ln1_b, out,
            n, 1e-5, 1.3209,
            BLOCK=1024,
            num_warps=8,
        )
        return out

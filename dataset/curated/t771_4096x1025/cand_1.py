import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 771
M, D, DT = 4096, 1025, torch.bfloat16


@triton.jit
def _fused_gelu_bias_softmax(X, B, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    ptr = X + row * N + offs

    # load matmul output (bf16) and upcast to fp32 (PyTorch opmath behavior)
    x = tl.load(ptr).to(tl.float32)

    # exact (erf-based) GELU in fp32, then round to bf16 like the reference intermediate
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # bias add in fp32, round to bf16 (reference stores bf16 before softmax)
    b = tl.load(B + offs).to(tl.float32)
    s = (g + b).to(tl.bfloat16).to(tl.float32)

    # numerically-stable softmax in fp32
    m = tl.max(s, 0)
    e = tl.exp(s - m)
    denom = tl.sum(e, 0)
    out = e / denom

    tl.store(Y + row * N + offs, out.to(tl.bfloat16))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 1024, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores, fp32 accumulate) — same as reference
        h = x @ self.W0

        if not h.is_cuda:
            # CPU fallback: original path
            h = F.gelu(h)
            h = h + self.b2
            return torch.softmax(h, dim=-1)

        h = h.contiguous()
        rows, N = h.shape
        y = torch.empty_like(h)
        _fused_gelu_bias_softmax[(rows,)](
            h, self.b2, y,
            N=N, BLOCK=N,
            num_warps=8,
        )
        return y

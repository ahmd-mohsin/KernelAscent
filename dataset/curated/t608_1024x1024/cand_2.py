import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 608
M, D, DT = 1024, 1024, torch.bfloat16


@triton.jit
def _ln_gelu_softmax_kernel(
    X_ptr, G_ptr, B_ptr, Out_ptr,
    N, stride,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride + offs, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm (fp32 accumulation, like PyTorch)
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g + b
    # layer_norm output is bf16 in the reference chain
    y = y.to(tl.bfloat16).to(tl.float32)

    # Exact (erf-based) GELU
    ge = y * 0.5 * (1.0 + tl.math.erf(y * 0.7071067811865476))
    # gelu output cast back to bf16 before softmax (as in reference)
    ge = ge.to(tl.bfloat16).to(tl.float32)

    # Softmax (fp32 accumulation)
    ge = tl.where(mask, ge, float('-inf'))
    m = tl.max(ge, axis=0)
    e = tl.exp(ge - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Out_ptr + row * stride + offs, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W1 = nn.Parameter((torch.randn(1024, 4096, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x * 1.1627
        h = x @ self.W1  # cuBLAS bf16 GEMM (tensor cores)
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _ln_gelu_softmax_kernel[(Mrows,)](
            h, self.ln2_g, self.ln2_b, out,
            N, h.stride(0),
            1e-5,
            BLOCK=BLOCK,
            num_warps=16,
        )
        return out

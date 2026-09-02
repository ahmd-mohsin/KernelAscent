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
    X, G, B, Out,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * N + offs, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm (fp32 math, like PyTorch's internal computation)
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    y = diff * rstd * g + b
    # match PyTorch: layer_norm output cast back to bf16
    y = y.to(tl.bfloat16).to(tl.float32)

    # GELU (erf variant, fp32 opmath, output rounded to bf16)
    ge = 0.5 * y * (1.0 + tl.math.erf(y * 0.7071067811865476))
    ge = ge.to(tl.bfloat16).to(tl.float32)

    # Softmax (fp32 accumulation)
    ge = tl.where(mask, ge, float('-inf'))
    row_max = tl.max(ge, axis=0)
    e = tl.exp(ge - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(Out + row * N + offs, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W1 = nn.Parameter((torch.randn(1024, 4096, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # scale + matmul (tensor-core GEMM in bf16, same as reference)
        h = torch.matmul(x * 1.1627, self.W1)
        h = h.contiguous()
        rows, N = h.shape[0], h.shape[-1]
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _ln_gelu_softmax_kernel[(rows,)](
            h, self.ln2_g, self.ln2_b, out,
            N, 1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out

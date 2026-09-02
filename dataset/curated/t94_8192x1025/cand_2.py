import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 94
M, D, DT = 8192, 1025, torch.bfloat16


@triton.jit
def _fused_act_bias_softmax(
    X_ptr, B_ptr, Y_ptr,
    stride_x, stride_y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # load matmul output (bf16), upcast to fp32 for math
    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # relu
    x = tl.maximum(x, 0.0)

    # exact (erf) gelu computed in fp32, rounded back to bf16 (matches PyTorch bf16 gelu)
    g = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # relu (no-op numerically after gelu of non-negative input, kept for equivalence)
    g = tl.maximum(g, 0.0)

    # bias add in fp32, rounded to bf16 (matches PyTorch bf16 add semantics)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    v = (g + b).to(tl.bfloat16).to(tl.float32)

    # softmax in fp32 (matches PyTorch internal accumulation), masked positions -> -inf
    v = tl.where(mask, v, float('-inf'))
    m = tl.max(v, 0)
    e = tl.exp(v - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, 0)
    out = e / s

    tl.store(Y_ptr + row * stride_y + offs, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 512, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback: reference path
            h = x @ self.W0
            h = torch.relu(h)
            h = F.gelu(h)
            h = torch.relu(h)
            h = h + self.b4
            return torch.softmax(h, dim=-1)

        # cuBLAS bf16 matmul (tensor cores)
        h = x @ self.W0
        rows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_act_bias_softmax[(rows,)](
            h, self.b4, out,
            h.stride(0), out.stride(0),
            N=N, BLOCK=BLOCK,
            num_warps=4,
        )
        return out

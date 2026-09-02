import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 954
M, D, DT = 1024, 512, torch.float16


@triton.jit
def _fused_ln_softmax_kernel(
    X_ptr, G_ptr, B_ptr, Y_ptr,
    N, eps, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    x = x * scale

    # LayerNorm (fp32 accumulation, biased variance, matching PyTorch)
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g + b

    # Round to fp16 (as PyTorch would materialize LN output) then softmax in fp32
    y = y.to(tl.float16).to(tl.float32)
    y = tl.where(mask, y, float('-inf'))
    m = tl.max(y, axis=0)
    e = tl.exp(y - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    # softmax output is >= 0, so ReLU is identity
    tl.store(Y_ptr + row * N + offs, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS half GEMM (tensor cores on A100)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        M_rows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_ln_softmax_kernel[(M_rows,)](
            h, self.ln2_g, self.ln2_b, out,
            N, 1e-5, 1.0315,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out

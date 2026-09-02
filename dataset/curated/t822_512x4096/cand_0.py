import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 822
M, D, DT = 512, 4096, torch.bfloat16


@triton.jit
def _fused_gelu_ln_relu_ln(
    X, G2, B2, G4, B4, Y,
    N: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # exact (erf) GELU, computed in fp32 then rounded to bf16 (match eager)
    x = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    x = x.to(tl.bfloat16).to(tl.float32)

    # LayerNorm 1 (stats in fp32)
    mean = tl.sum(tl.where(mask, x, 0.0)) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d) / N
    rstd = 1.0 / tl.sqrt(var + EPS)
    g2 = tl.load(G2 + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    x = d * rstd * g2 + b2
    x = x.to(tl.bfloat16).to(tl.float32)

    # ReLU
    x = tl.maximum(x, 0.0)

    # LayerNorm 2
    mean2 = tl.sum(tl.where(mask, x, 0.0)) / N
    d2 = tl.where(mask, x - mean2, 0.0)
    var2 = tl.sum(d2 * d2) / N
    rstd2 = 1.0 / tl.sqrt(var2 + EPS)
    g4 = tl.load(G4 + cols, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(B4 + cols, mask=mask, other=0.0).to(tl.float32)
    y = d2 * rstd2 * g4 + b4

    tl.store(Y + row * N + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            h = x @ self.W0
            h = F.gelu(h)
            h = F.layer_norm(h, (h.shape[-1],), self.ln2_g, self.ln2_b)
            h = torch.relu(h)
            h = F.layer_norm(h, (h.shape[-1],), self.ln4_g, self.ln4_b)
            return h

        # tensor-core GEMM (cuBLAS)
        h = torch.matmul(x, self.W0)

        orig_shape = h.shape
        N = orig_shape[-1]
        h2 = h.contiguous().view(-1, N)
        rows = h2.shape[0]
        y = torch.empty_like(h2)

        BLOCK = triton.next_power_of_2(N)
        _fused_gelu_ln_relu_ln[(rows,)](
            h2, self.ln2_g, self.ln2_b, self.ln4_g, self.ln4_b, y,
            N=N, EPS=1e-5, BLOCK=BLOCK,
            num_warps=4,
        )
        return y.view(orig_shape)

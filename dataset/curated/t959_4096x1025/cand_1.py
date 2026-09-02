import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 959
M, D, DT = 4096, 1025, torch.bfloat16


@triton.jit
def _fused_ln_ln_bias_relu_softmax(
    X, G1, B1, G2, B2, B3, Out,
    N: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * N + offs, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm 1 (fp32 accumulation, round output to bf16 like PyTorch)
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    inv = 1.0 / tl.sqrt(var + EPS)
    g1 = tl.load(G1 + offs, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + offs, mask=mask, other=0.0).to(tl.float32)
    x = d * inv * g1 + b1
    x = x.to(tl.bfloat16).to(tl.float32)

    # LayerNorm 2
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    inv = 1.0 / tl.sqrt(var + EPS)
    g2 = tl.load(G2 + offs, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + offs, mask=mask, other=0.0).to(tl.float32)
    x = d * inv * g2 + b2
    x = x.to(tl.bfloat16).to(tl.float32)

    # bias add (rounded to bf16 like a bf16 elementwise add) + relu
    b3 = tl.load(B3 + offs, mask=mask, other=0.0).to(tl.float32)
    x = (x + b3).to(tl.bfloat16).to(tl.float32)
    x = tl.maximum(x, 0.0)

    # softmax (fp32 accumulation)
    x = tl.where(mask, x, float('-inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(Out + row * N + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 512, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (bf16 tensor cores)
        h = torch.matmul(x, self.W0)
        if not h.is_cuda:
            h = F.layer_norm(h, (h.shape[-1],), self.ln1_g, self.ln1_b)
            h = F.layer_norm(h, (h.shape[-1],), self.ln2_g, self.ln2_b)
            h = torch.relu(h + self.b3)
            return torch.softmax(h, dim=-1)

        h = h.contiguous()
        rows, N = h.shape[0], h.shape[1]
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_ln_ln_bias_relu_softmax[(rows,)](
            h, self.ln1_g, self.ln1_b, self.ln2_g, self.ln2_b, self.b3, out,
            N=N, EPS=1e-5, BLOCK=BLOCK,
            num_warps=4,
        )
        return out

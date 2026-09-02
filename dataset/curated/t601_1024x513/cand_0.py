import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 601
M, D, DT = 1024, 513, torch.bfloat16


@triton.jit
def _fused_relu_ln_softmax_ln_kernel(
    X, G2, B2, G4, B4, OUT,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # load row (bf16 -> fp32), relu (exact on bf16, no rounding issue)
    x = tl.load(X + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    x = tl.maximum(x, 0.0)

    # ---- LayerNorm 1 (fp32 math, like PyTorch on bf16 input) ----
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g2 = tl.load(G2 + offs, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + offs, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g2 + b2
    # cast back to bf16 (output dtype of layer_norm), then re-upcast
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- Softmax (fp32 accumulation, bf16 output) ----
    y = tl.where(mask, y, float('-inf'))
    m = tl.max(y, axis=0)
    e = tl.exp(y - m)
    e = tl.where(mask, e, 0.0)
    s = e / tl.sum(e, axis=0)
    s = s.to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm 2 ----
    mean2 = tl.sum(s, axis=0) / N
    d2 = tl.where(mask, s - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + 1e-5)
    g4 = tl.load(G4 + offs, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(B4 + offs, mask=mask, other=0.0).to(tl.float32)
    z = d2 * rstd2 * g4 + b4
    # layer_norm output cast to bf16, then scalar mul in fp32, cast to bf16
    z = z.to(tl.bfloat16).to(tl.float32) * 1.2373

    tl.store(OUT + row * N + offs, z.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 512, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # tensor-core bf16 GEMM
        orig_shape = h.shape
        N = orig_shape[-1]
        h2 = h.contiguous().view(-1, N)
        rows = h2.shape[0]
        out = torch.empty_like(h2)
        BLOCK = triton.next_power_of_2(N)
        _fused_relu_ln_softmax_ln_kernel[(rows,)](
            h2, self.ln2_g, self.ln2_b, self.ln4_g, self.ln4_b, out,
            N=N, BLOCK=BLOCK,
            num_warps=4,
        )
        return out.view(orig_shape)

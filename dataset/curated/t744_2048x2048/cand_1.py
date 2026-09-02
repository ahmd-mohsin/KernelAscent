import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 744
M, D, DT = 2048, 2048, torch.float16


@triton.jit
def _fused_relu_scale_ln_ln(
    X, G3, B3, G4, B4, Y,
    N, eps, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # relu + scale (opmath fp32, then round to fp16 to match reference intermediate)
    x = tl.maximum(x, 0.0) * scale
    x = x.to(tl.float16).to(tl.float32)

    # LayerNorm 1 (stats in fp32, like PyTorch)
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    g3 = tl.load(G3 + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    x = d * rstd * g3 + b3
    # round to fp16 to match reference intermediate output of layer_norm
    x = x.to(tl.float16).to(tl.float32)

    # LayerNorm 2
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    g4 = tl.load(G4 + cols, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(B4 + cols, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g4 + b4

    tl.store(Y + row * N + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS tensor cores
        h = x @ self.W0

        if not h.is_cuda:
            h = torch.relu(h) * 1.0687
            h = F.layer_norm(h, (h.shape[-1],), self.ln3_g, self.ln3_b)
            h = F.layer_norm(h, (h.shape[-1],), self.ln4_g, self.ln4_b)
            return h

        orig_shape = h.shape
        N = orig_shape[-1]
        h2 = h.contiguous().view(-1, N)
        rows = h2.shape[0]
        out = torch.empty_like(h2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_relu_scale_ln_ln[(rows,)](
            h2, self.ln3_g, self.ln3_b, self.ln4_g, self.ln4_b, out,
            N, 1e-5, 1.0687,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 37
M, D, DT = 8192, 512, torch.float16


@triton.jit
def _fused_gelu_relu_ln_scale(
    X, G, B, Y,
    N, eps, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # exact (erf) GELU in fp32, then cast to fp16 (matches PyTorch fp16 gelu opmath)
    g = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.float16)

    # ReLU
    r = tl.maximum(g, 0.0).to(tl.float16)

    # LayerNorm in fp32
    rf = r.to(tl.float32)
    rf = tl.where(mask, rf, 0.0)
    mean = tl.sum(rf, axis=0) / N
    diff = tl.where(mask, rf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    gamma = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = (rf - mean) * rstd * gamma + beta
    y16 = y.to(tl.float16)

    # scale by 1.4798 (fp32 opmath, fp16 result — matches PyTorch scalar mul)
    out = (y16.to(tl.float32) * scale).to(tl.float16)

    tl.store(Y + row * N + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS fp16 tensor-core matmul
        h = h.contiguous()
        rows, N = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_gelu_relu_ln_scale[(rows,)](
            h, self.ln3_g, self.ln3_b, y,
            N, 1e-5, 1.4798,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return y

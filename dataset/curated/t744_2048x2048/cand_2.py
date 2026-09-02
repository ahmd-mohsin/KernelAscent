import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 744
M, D, DT = 2048, 2048, torch.float16


@triton.jit
def _fused_relu_scale_dual_ln(
    X, G3, B3, G4, B4, Y,
    N, stride,
    scale, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride + offs, mask=mask, other=0.0).to(tl.float32)

    # relu + scale (match fp16 rounding of the reference's fp16 multiply)
    x = tl.maximum(x, 0.0) * scale
    x = x.to(tl.float16).to(tl.float32)

    # --- LayerNorm 1 (fp32 accumulation, like PyTorch's fp16 layer_norm) ---
    mean1 = tl.sum(tl.where(mask, x, 0.0), axis=0) / N
    d1 = tl.where(mask, x - mean1, 0.0)
    var1 = tl.sum(d1 * d1, axis=0) / N
    rstd1 = 1.0 / tl.sqrt(var1 + eps)

    g3 = tl.load(G3 + offs, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + offs, mask=mask, other=0.0).to(tl.float32)
    z = d1 * rstd1 * g3 + b3
    # reference materializes fp16 between the two layer norms
    z = z.to(tl.float16).to(tl.float32)

    # --- LayerNorm 2 ---
    mean2 = tl.sum(tl.where(mask, z, 0.0), axis=0) / N
    d2 = tl.where(mask, z - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + eps)

    g4 = tl.load(G4 + offs, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(B4 + offs, mask=mask, other=0.0).to(tl.float32)
    y = d2 * rstd2 * g4 + b4

    tl.store(Y + row * stride + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        # scale as it would be rounded to fp16 (matches fp16 tensor * python float)
        self._scale = torch.tensor(1.0687, dtype=torch.float16).item()

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores)
        h = x @ self.W0
        h = h.contiguous()
        M_, N_ = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N_)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_relu_scale_dual_ln[(M_,)](
            h, self.ln3_g, self.ln3_b, self.ln4_g, self.ln4_b, out,
            N_, h.stride(0),
            self._scale, 1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out

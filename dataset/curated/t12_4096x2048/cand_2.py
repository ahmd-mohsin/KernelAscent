import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 12
M, D, DT = 4096, 2048, torch.bfloat16


@triton.jit
def _double_ln_scale_kernel(
    X, OUT,
    G1, B1, G2, B2,
    N, stride_x, stride_o,
    scale, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # --- LayerNorm 1 (fp32 math, matching PyTorch's internal accumulation) ---
    mean1 = tl.sum(x, axis=0) / N
    d1 = tl.where(mask, x - mean1, 0.0)
    var1 = tl.sum(d1 * d1, axis=0) / N
    rstd1 = 1.0 / tl.sqrt(var1 + eps)

    g1 = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    y1 = d1 * rstd1 * g1 + b1

    # round intermediate to bf16 (reference materializes bf16 between LNs)
    y1 = y1.to(tl.bfloat16).to(tl.float32)
    y1 = tl.where(mask, y1, 0.0)

    # --- LayerNorm 2 ---
    mean2 = tl.sum(y1, axis=0) / N
    d2 = tl.where(mask, y1 - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + eps)

    g2 = tl.load(G2 + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    y2 = d2 * rstd2 * g2 + b2

    # reference casts to bf16 after LN2, then multiplies by scalar (fp32 opmath)
    y2 = y2.to(tl.bfloat16).to(tl.float32)
    out = (y2 * scale).to(tl.bfloat16)

    tl.store(OUT + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS matmul (tensor cores)
        y = x @ self.W0

        if not y.is_cuda:
            y = F.layer_norm(y, (y.shape[-1],), self.ln1_g, self.ln1_b)
            y = F.layer_norm(y, (y.shape[-1],), self.ln2_g, self.ln2_b)
            return y * 1.2791

        orig_shape = y.shape
        N = orig_shape[-1]
        y2d = y.contiguous().view(-1, N)
        rows = y2d.shape[0]
        out = torch.empty_like(y2d)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _double_ln_scale_kernel[(rows,)](
            y2d, out,
            self.ln1_g, self.ln1_b, self.ln2_g, self.ln2_b,
            N, y2d.stride(0), out.stride(0),
            1.2791, 1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)

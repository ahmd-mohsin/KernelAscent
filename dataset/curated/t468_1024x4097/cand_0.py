import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 468
M, D, DT = 1024, 4097, torch.float16


@triton.jit
def _fused_scale_double_ln(
    X, G1, B1, G2, B2, Y,
    D, stride_x, stride_y,
    scale, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    # replicate: fp16 tensor = (x.f32 * scale) rounded to fp16
    t = (x * scale).to(tl.float16).to(tl.float32)

    # ---- LayerNorm 1 (stats in fp32) ----
    mean1 = tl.sum(t, axis=0) / D
    d1 = tl.where(mask, t - mean1, 0.0)
    var1 = tl.sum(d1 * d1, axis=0) / D
    rstd1 = 1.0 / tl.sqrt(var1 + eps)

    g1 = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    # intermediate is materialized as fp16 in reference -> round to fp16
    y1 = (d1 * rstd1 * g1 + b1).to(tl.float16).to(tl.float32)
    y1 = tl.where(mask, y1, 0.0)

    # ---- LayerNorm 2 ----
    mean2 = tl.sum(y1, axis=0) / D
    d2 = tl.where(mask, y1 - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / D
    rstd2 = 1.0 / tl.sqrt(var2 + eps)

    g2 = tl.load(G2 + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    out = d2 * rstd2 * g2 + b2

    tl.store(Y + row * stride_y + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x * 1.3642
            x = F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)
            x = F.layer_norm(x, (x.shape[-1],), self.ln2_g, self.ln2_b)
            return x

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK <= 4096 else 16

        _fused_scale_double_ln[(m,)](
            x2, self.ln1_g, self.ln1_b, self.ln2_g, self.ln2_b, y,
            d, x2.stride(0), y.stride(0),
            1.3642, 1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)

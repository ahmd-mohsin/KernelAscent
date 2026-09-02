import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 433
M, D, DT = 1024, 4097, torch.bfloat16


@triton.jit
def _fused_rms_softmax_gelu_kernel(
    X, W, Y,
    D,
    stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    # ---- load row in fp32 ----
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- RMSNorm (fp32 math, round to bf16, mul by bf16 weight in fp32 opmath) ----
    ms = tl.sum(x * x, axis=0) / D
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    xn = (x * inv).to(tl.bfloat16).to(tl.float32)

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    v = (xn * w).to(tl.bfloat16).to(tl.float32)

    # ---- softmax (fp32 accumulation like PyTorch, output rounded to bf16) ----
    v_masked = tl.where(mask, v, float('-inf'))
    row_max = tl.max(v_masked, axis=0)
    e = tl.math.exp(v - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    p = (e / denom).to(tl.bfloat16).to(tl.float32)

    # ---- exact GELU (erf) in fp32 opmath, round to bf16 ----
    g = 0.5 * p * (1.0 + tl.math.erf(p * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # ---- scale by 1.158 (fp32 opmath, round to bf16) ----
    y = (g * 1.158).to(tl.bfloat16)

    tl.store(Y + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            x = torch.softmax(x, dim=-1)
            x = F.gelu(x)
            return x * 1.158

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK <= 4096 else 16

        _fused_rms_softmax_gelu_kernel[(m,)](
            x2, self.rms0_w, y,
            d,
            x2.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)

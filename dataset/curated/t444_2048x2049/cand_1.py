import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 444
M, D, DT = 2048, 2049, torch.bfloat16


@triton.jit
def _fused_ln_softmax_kernel(
    X, G, B, B2, Y,
    N, eps, scale,
    stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm (fp32 math) ----
    n = N.to(tl.float32)
    mean = tl.sum(x, axis=0) / n
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / n
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    ln = (x - mean) * rstd * g + b

    # round to bf16 (layer_norm output dtype), then upcast for softmax
    ln = ln.to(tl.bfloat16).to(tl.float32)

    # ---- Softmax (fp32 math) ----
    ln_masked = tl.where(mask, ln, float("-inf"))
    row_max = tl.max(ln_masked, axis=0)
    e = tl.exp(ln_masked - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    sm = e / denom

    # round to bf16 (softmax output dtype)
    sm = sm.to(tl.bfloat16).to(tl.float32)

    # ---- add bias (bf16 tensors, fp32 opmath, round back) ----
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    out = (sm + b2)
    out = out.to(tl.bfloat16).to(tl.float32)

    # ---- scale ----
    out = out * scale

    tl.store(Y + row * stride_y + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
            y = torch.softmax(y, dim=-1)
            return (y + self.b2) * 1.3125

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_ln_softmax_kernel[(rows,)](
            x2, self.ln0_g, self.ln0_b, self.b2, y,
            N, 1e-5, 1.3125,
            x2.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)

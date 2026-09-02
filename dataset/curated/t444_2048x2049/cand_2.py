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
    N, stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x_ptr = X + row * stride_x + cols
    x = tl.load(x_ptr, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm (fp32 accumulation, like PyTorch) ----
    n_f = N.to(tl.float32)
    mean = tl.sum(x, axis=0) / n_f
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / n_f
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    ln = (x - mean) * rstd * g + b
    # round to bf16 (matches intermediate materialization in reference)
    ln = ln.to(tl.bfloat16).to(tl.float32)

    # ---- Softmax (fp32 accumulation) ----
    ln_masked = tl.where(mask, ln, float("-inf"))
    row_max = tl.max(ln_masked, axis=0)
    e = tl.exp(ln_masked - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    sm = e / denom
    sm = sm.to(tl.bfloat16).to(tl.float32)

    # ---- add bias, scale (bf16 semantics: round after each op) ----
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    out = (sm + b2).to(tl.bfloat16).to(tl.float32)
    out = (out * 1.3125).to(tl.bfloat16)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_ln_softmax_kernel[(rows,)](
            x2, self.ln0_g, self.ln0_b, self.b2, y,
            N, x2.stride(0), y.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view(orig_shape)

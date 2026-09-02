import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 942
M, D, DT = 4096, 1024, torch.float16


@triton.jit
def _fused_kernel(
    X, W0, G, B, Y,
    stride_x, stride_y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- RMSNorm (fp32 math, round to fp16, then fp16-rounded weight mul) ----
    ms = tl.sum(x * x, axis=0) / N
    rstd0 = 1.0 / tl.sqrt(ms + 1e-6)
    xn = (x * rstd0).to(tl.float16)

    w0 = tl.load(W0 + cols, mask=mask, other=0.0)
    h = (xn.to(tl.float32) * w0.to(tl.float32)).to(tl.float16)

    # ---- ReLU ----
    h = tl.maximum(h, 0.0)

    # ---- LayerNorm (fp32 accumulate) ----
    hf = tl.where(mask, h.to(tl.float32), 0.0)
    mean = tl.sum(hf, axis=0) / N
    diff = tl.where(mask, hf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    ln = ((hf - mean) * rstd * g + b).to(tl.float16)

    # ---- GELU (exact, erf) ----
    lf = ln.to(tl.float32)
    out = 0.5 * lf * (1.0 + tl.math.erf(lf * 0.7071067811865476))

    tl.store(Y + row * stride_y + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        Mrows = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 1024 else 4
        _fused_kernel[(Mrows,)](
            x2, self.rms0_w, self.ln2_g, self.ln2_b, y,
            x2.stride(0), y.stride(0),
            N=N, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 439
M, D, DT = 4096, 2048, torch.bfloat16


@triton.jit
def _fused_kernel(
    X, G, B, Y,
    stride_xm, stride_ym,
    N, eps, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)

    # exact (erf-based) GELU, then cast to bf16 to match reference intermediate dtype
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g_out = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    g_out = g_out.to(tl.bfloat16).to(tl.float32)

    # LayerNorm (fp32 accumulation like PyTorch), output cast to bf16
    mean = tl.sum(tl.where(mask, g_out, 0.0), axis=0) / N
    diff = tl.where(mask, g_out - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    bb = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (g_out - mean) * rstd * w + bb
    y = y.to(tl.bfloat16).to(tl.float32)

    # ReLU
    y = tl.maximum(y, 0.0)

    # Softmax in fp32, output bf16
    y = tl.where(mask, y, float("-inf"))
    m = tl.max(y, axis=0)
    e = tl.exp(y - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = (e / s).to(tl.bfloat16).to(tl.float32)

    # scale (fp32 opmath, cast back to bf16)
    out = (sm * scale).to(tl.bfloat16)
    tl.store(Y + row * stride_ym + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = F.gelu(x)
            y = F.layer_norm(y, (y.shape[-1],), self.ln1_g, self.ln1_b)
            y = torch.relu(y)
            y = torch.softmax(y, dim=-1)
            return y * 1.4972

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        Mrows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_kernel[(Mrows,)](
            x2, self.ln1_g, self.ln1_b, out,
            x2.stride(0), out.stride(0),
            N, 1e-5, 1.4972,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)

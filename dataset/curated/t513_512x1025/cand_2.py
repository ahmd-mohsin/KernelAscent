import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 513
M, D, DT = 512, 1025, torch.float16


@triton.jit
def _fused_kernel(
    X, G, B, B3, Y,
    N, stride_x, stride_y,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # GELU (exact, erf-based), computed then rounded to fp16 like PyTorch fp16 op
    g = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.float16).to(tl.float32)

    # Softmax 1
    g_m = tl.where(mask, g, float('-inf'))
    mx = tl.max(g_m, axis=0)
    e = tl.exp(g_m - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm1 = (e / s).to(tl.float16).to(tl.float32)

    # LayerNorm
    mean = tl.sum(tl.where(mask, sm1, 0.0), axis=0) / N
    d = tl.where(mask, sm1 - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)
    gamma = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    ln = (d * rstd) * gamma + beta
    ln = ln.to(tl.float16).to(tl.float32)

    # Add bias (fp16 add semantics)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    a = (ln + b3)
    a = a.to(tl.float16).to(tl.float32)

    # Softmax 2
    a_m = tl.where(mask, a, float('-inf'))
    mx2 = tl.max(a_m, axis=0)
    e2 = tl.exp(a_m - mx2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    out = e2 / s2

    tl.store(Y + row * stride_y + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln2_g = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = F.gelu(x)
            y = torch.softmax(y, dim=-1)
            y = F.layer_norm(y, (y.shape[-1],), self.ln2_g, self.ln2_b)
            y = y + self.b3
            return torch.softmax(y, dim=-1)

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_kernel[(rows,)](
            x2, self.ln2_g, self.ln2_b, self.b3, y,
            N, x2.stride(0), y.stride(0),
            EPS=1e-5, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)

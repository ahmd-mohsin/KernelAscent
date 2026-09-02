import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 466
M, D, DT = 512, 513, torch.float16


@triton.jit
def _ln_scale_bias_kernel(
    X, Y, G, B, B3,
    N,
    stride_x, stride_y,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    ln = (diff * rstd) * g + b
    # round to fp16 like F.layer_norm output, then do fp16 scale + fp16 add
    ln16 = ln.to(tl.float16)
    scale = tl.full((1,), 1.2009, tl.float16)
    y = ln16 * scale
    b3 = tl.load(B3 + cols, mask=mask, other=0.0)
    y = y + b3

    tl.store(Y + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 4096, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = torch.matmul(x, self.W0)
        if not h.is_cuda:
            h = F.layer_norm(h, (h.shape[-1],), self.ln1_g, self.ln1_b)
            return h * 1.2009 + self.b3

        h = h.contiguous()
        rows, N = h.shape[0], h.shape[-1]
        h2 = h.view(-1, N)
        out = torch.empty_like(h2)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _ln_scale_bias_kernel[(h2.shape[0],)](
            h2, out, self.ln1_g, self.ln1_b, self.b3,
            N, h2.stride(0), out.stride(0),
            EPS=1e-5, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view_as(h)

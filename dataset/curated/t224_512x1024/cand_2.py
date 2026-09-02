import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 224
M, D, DT = 512, 1024, torch.float16


@triton.jit
def _ln_scale_kernel(
    X, G, B, Y,
    N, eps, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = xc * rstd * g + b
    # match F.layer_norm output rounding to fp16, then fp16-tensor * float-scalar
    # (PyTorch computes half elementwise mul in float, then rounds)
    y16 = y.to(tl.float16)
    out = (y16.to(tl.float32) * scale).to(tl.float16)
    tl.store(Y + row * N + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 1024, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0
        if not h.is_cuda:
            h = F.layer_norm(h, (h.shape[-1],), self.ln1_g, self.ln1_b)
            return h * 1.052
        h = h.contiguous()
        Mrows, N = h.shape[-2] if h.dim() > 1 else 1, h.shape[-1]
        h2 = h.view(-1, N)
        rows = h2.shape[0]
        out = torch.empty_like(h2)
        BLOCK = triton.next_power_of_2(N)
        _ln_scale_kernel[(rows,)](
            h2, self.ln1_g, self.ln1_b, out,
            N, 1e-5, 1.052,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out.view(h.shape)

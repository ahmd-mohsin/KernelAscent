import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 40
M, D, DT = 4096, 513, torch.float16


@triton.jit
def _fused_ln_add_ln_kernel(
    X, Y, G0, B0, B1, G2, B2,
    N, stride_x, stride_y,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # First LayerNorm (fp32 accumulation, like PyTorch on fp16 CUDA)
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    g0 = tl.load(G0 + cols, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(B0 + cols, mask=mask, other=0.0).to(tl.float32)
    y = diff * rstd * g0 + b0

    # Round to fp16 (matches intermediate fp16 storage in reference)
    y16 = y.to(tl.float16)

    # Add bias in fp16 arithmetic (matches x + self.b1 in fp16)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0)
    z16 = y16 + b1
    z = z16.to(tl.float32)
    z = tl.where(mask, z, 0.0)

    # Second LayerNorm
    mean2 = tl.sum(z, axis=0) / N
    diff2 = tl.where(mask, z - mean2, 0.0)
    var2 = tl.sum(diff2 * diff2, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + EPS)

    g2 = tl.load(G2 + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    out = diff2 * rstd2 * g2 + b2

    tl.store(Y + row * stride_y + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
            y = y + self.b1
            return F.layer_norm(y, (y.shape[-1],), self.ln2_g, self.ln2_b)

        orig_shape = x.shape
        N = orig_shape[-1]
        x2d = x.contiguous().view(-1, N)
        rows = x2d.shape[0]
        out = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 1024 else 8

        _fused_ln_add_ln_kernel[(rows,)](
            x2d, out,
            self.ln0_g, self.ln0_b, self.b1, self.ln2_g, self.ln2_b,
            N, x2d.stride(0), out.stride(0),
            EPS=1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)

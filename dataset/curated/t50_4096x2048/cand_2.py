import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 50
M, D, DT = 4096, 2048, torch.float16


@triton.jit
def _fused_kernel(X, OUT, G1, B1, G3, B3,
                  N, stride_x, stride_o,
                  S1, S2, EPS,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=-float('inf')).to(tl.float32)

    # softmax (fp32 accumulation, cast result to fp16 like PyTorch output)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = (e / s).to(tl.float16).to(tl.float32)

    # layernorm 1
    mean1 = tl.sum(p, axis=0) / N
    d1 = tl.where(mask, p - mean1, 0.0)
    var1 = tl.sum(d1 * d1, axis=0) / N
    rstd1 = 1.0 / tl.sqrt(var1 + EPS)
    g1 = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (d1 * rstd1) * g1 + b1
    y16 = y.to(tl.float16)
    # scale in fp16 (matches x * 1.1736 on fp16 tensor)
    y16 = y16 * S1.to(tl.float16)
    y = y16.to(tl.float32)

    # layernorm 2
    y = tl.where(mask, y, 0.0)
    mean2 = tl.sum(y, axis=0) / N
    d2 = tl.where(mask, y - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + EPS)
    g3 = tl.load(G3 + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    z = (d2 * rstd2) * g3 + b3
    z16 = z.to(tl.float16)
    z16 = z16 * S2.to(tl.float16)

    tl.store(OUT + row * stride_o + cols, z16, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = torch.softmax(x, dim=-1)
            y = F.layer_norm(y, (y.shape[-1],), self.ln1_g, self.ln1_b) * 1.1736
            y = F.layer_norm(y, (y.shape[-1],), self.ln3_g, self.ln3_b) * 1.0344
            return y

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        Mrows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_kernel[(Mrows,)](
            x2, out,
            self.ln1_g, self.ln1_b, self.ln3_g, self.ln3_b,
            N, x2.stride(0), out.stride(0),
            1.1736, 1.0344, 1e-5,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.view(orig_shape)

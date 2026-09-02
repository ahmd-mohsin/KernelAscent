import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 201
M, D, DT = 512, 2048, torch.bfloat16

_INV_SQRT2 = 0.7071067811865476


@triton.jit
def _fused_kernel(X, W, Y, N, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    x = tl.load(X + row * D + cols, mask=mask, other=0.0).to(tl.float32)

    # gelu (exact, erf) in fp32, round to bf16 (match PyTorch bf16 output)
    g = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16)

    # relu
    r = tl.maximum(g, 0.0).to(tl.bfloat16)

    # rmsnorm in fp32
    rf = r.to(tl.float32)
    ms = tl.sum(rf * rf, axis=0) / D
    inv = tl.math.rsqrt(ms + 1e-6)
    n = (rf * inv).to(tl.bfloat16)

    # multiply by weight (fp32 compute, bf16 round — matches PyTorch opmath)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    m = (n.to(tl.float32) * w).to(tl.bfloat16)

    # final gelu
    mf = m.to(tl.float32)
    out = mf * 0.5 * (1.0 + tl.math.erf(mf * 0.7071067811865476))
    out = out.to(tl.bfloat16)

    tl.store(Y + row * D + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = F.gelu(x)
            x = torch.relu(x)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            return F.gelu(x)

        orig_shape = x.shape
        Dm = orig_shape[-1]
        xc = x.contiguous().view(-1, Dm)
        n_rows = xc.shape[0]
        y = torch.empty_like(xc)
        BLOCK = triton.next_power_of_2(Dm)
        _fused_kernel[(n_rows,)](
            xc, self.rms2_w, y, n_rows, Dm, BLOCK=BLOCK,
            num_warps=8,
        )
        return y.view(orig_shape)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 899
M, D, DT = 1024, 512, torch.float16


@triton.jit
def _rms_softmax_kernel(
    X, W, Y,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    eps,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * D + offs, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(x * x, axis=0) / D
    inv = 1.0 / tl.sqrt(ms + eps)
    xn = (x * inv).to(tl.float16)

    # scale by weight in fp16 (matches fp16 * fp16 semantics)
    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float16)
    v = (xn * w).to(tl.float32)

    # softmax (fp32 accumulation, matching PyTorch half softmax)
    v = tl.where(mask, v, float('-inf'))
    m = tl.max(v, axis=0)
    e = tl.exp(v - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s

    # relu(relu(x)) is identity on softmax output (nonnegative)
    tl.store(Y + row * D + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            x = torch.softmax(x, dim=-1)
            return torch.relu(torch.relu(x))

        orig_shape = x.shape
        d = orig_shape[-1]
        xc = x.contiguous().view(-1, d)
        n_rows = xc.shape[0]
        y = torch.empty_like(xc)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 4 if BLOCK <= 1024 else 8

        _rms_softmax_kernel[(n_rows,)](
            xc, self.rms0_w, y,
            D=d, BLOCK=BLOCK, eps=1e-6,
            num_warps=num_warps,
        )
        return y.view(orig_shape)

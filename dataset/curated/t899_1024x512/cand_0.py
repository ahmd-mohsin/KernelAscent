import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 899
M, D, DT = 1024, 512, torch.float16


@triton.jit
def _rms_softmax_relu_kernel(
    X_ptr, W_ptr, Y_ptr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    # ---- RMSNorm (computed in fp32, matching x.float() path) ----
    x = tl.load(X_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / D
    inv = tl.math.rsqrt(ms + EPS)
    xn = (x * inv).to(tl.float16)  # cast back to fp16 like .to(x.dtype)

    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float16)
    y16 = xn * w  # fp16 * fp16 elementwise (matches PyTorch type promotion)

    # ---- Softmax (fp16 input, fp32 accumulation, matching PyTorch) ----
    y = y16.to(tl.float32)
    y = tl.where(mask, y, float('-inf'))
    m = tl.max(y, axis=0)
    e = tl.exp(y - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    # relu(relu(softmax)) == softmax (outputs are non-negative); identity.
    tl.store(Y_ptr + row * D + offs, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            _xf = x.float()
            xr = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            xr = torch.softmax(xr, dim=-1)
            return torch.relu(torch.relu(xr))

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        n_rows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 4 if BLOCK <= 1024 else 8

        _rms_softmax_relu_kernel[(n_rows,)](
            x2, self.rms0_w, y,
            D=d, EPS=1e-6, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)

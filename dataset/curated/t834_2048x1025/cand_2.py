import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 834
M, D, DT = 2048, 1025, torch.float16


@triton.jit
def _relu_rmsnorm_kernel(
    X, W, Y,
    D_len,
    stride_xm, stride_ym,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D_len

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    # relu
    x = tl.maximum(x, 0.0)
    xf = x.to(tl.float32)

    # mean of squares in fp32
    ss = tl.sum(xf * xf, axis=0) / D_len
    inv = 1.0 / tl.sqrt(ss + eps)

    normed = (xf * inv).to(Y.dtype.element_ty)  # cast to fp16 first (matches .to(x.dtype))
    w = tl.load(W + cols, mask=mask, other=0.0)
    out = normed * w  # fp16 * fp16 multiply, as in reference

    tl.store(Y + row * stride_ym + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = torch.relu(x)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            return x

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 2048 else 4

        _relu_rmsnorm_kernel[(m,)](
            x2, self.rms1_w, y,
            d,
            x2.stride(0), y.stride(0),
            1e-6,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)

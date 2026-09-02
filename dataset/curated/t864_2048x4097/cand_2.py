import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 864
M, D, DT = 2048, 4097, torch.float16


@triton.jit
def _fused_relu_rms_relu_kernel(
    X_ptr, W_ptr, Y_ptr,
    N,
    stride_x, stride_y,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0)
    xf = x.to(tl.float32)
    # relu
    xf = tl.maximum(xf, 0.0)

    # mean of squares over the row
    ms = tl.sum(xf * xf, axis=0) / N
    rs = 1.0 / tl.sqrt(ms + EPS)

    w = tl.load(W_ptr + offs, mask=mask, other=0.0)

    # normalize in fp32, cast back to fp16, multiply by weight (fp16), relu
    y = (xf * rs).to(tl.float16) * w
    y = tl.maximum(y, y * 0)

    tl.store(Y_ptr + row * stride_y + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if x.is_cuda and x.dtype == torch.float16:
            orig_shape = x.shape
            N = orig_shape[-1]
            x2d = x.contiguous().view(-1, N)
            rows = x2d.shape[0]
            y = torch.empty_like(x2d)
            w = self.rms1_w
            if w.device != x.device:
                w = w.to(x.device)
            BLOCK = triton.next_power_of_2(N)
            _fused_relu_rms_relu_kernel[(rows,)](
                x2d, w, y,
                N,
                x2d.stride(0), y.stride(0),
                EPS=1e-6,
                BLOCK=BLOCK,
                num_warps=8,
            )
            return y.view(orig_shape)

        # fallback (numerically identical reference path)
        x = torch.relu(x)
        _xf = x.float()
        x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
        x = torch.relu(x)
        return x

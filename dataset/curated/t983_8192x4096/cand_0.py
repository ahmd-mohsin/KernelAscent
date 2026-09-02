import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 983
M, D, DT = 8192, 4096, torch.bfloat16


@triton.jit
def _rmsnorm_relu_kernel(
    X, W, Y,
    stride_xm, stride_ym,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    mean_sq = tl.sum(xf * xf, axis=0) / N
    inv = tl.math.rsqrt(mean_sq + eps)

    # normalize in fp32, round to bf16 (matches .to(x.dtype))
    xn = (xf * inv).to(tl.bfloat16)

    w = tl.load(W + cols, mask=mask, other=0.0)
    # bf16 * bf16 elementwise with fp32 opmath, rounded to bf16 (matches PyTorch)
    y = (xn.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    # relu
    zero = tl.zeros_like(y)
    y = tl.where(y > zero, y, zero)

    tl.store(Y + row * stride_ym + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            return torch.relu(x)

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        Mrows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 4096 else 4

        _rmsnorm_relu_kernel[(Mrows,)](
            x2, self.rms0_w, y,
            x2.stride(0), y.stride(0),
            N, 1e-6,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)


def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(M, D, generator=g).to(DT)]

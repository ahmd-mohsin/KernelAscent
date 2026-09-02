import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 88
M, D, DT = 4096, 1024, torch.float16


@triton.jit
def _rmsnorm_gelu_kernel(
    X, W, Y,
    stride_x, stride_y,
    D_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D_

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # RMS norm in fp32
    ms = tl.sum(x * x, axis=0) / D_
    rrms = 1.0 / tl.sqrt(ms + 1e-6)
    xn = x * rrms

    # cast to fp16 (matching .to(x.dtype)), multiply by fp16 weight in fp16
    xn_h = xn.to(tl.float16)
    w = tl.load(W + cols, mask=mask, other=0.0)  # fp16
    h = xn_h * w  # fp16 arithmetic

    # gelu (exact, erf-based) computed in fp32 like PyTorch does for half inputs
    hf = h.to(tl.float32)
    out = 0.5 * hf * (1.0 + tl.math.erf(hf * 0.7071067811865476))

    tl.store(Y + row * stride_y + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        assert x.is_cuda, "expected CUDA tensor"
        x2 = x.contiguous()
        M_, D_ = x2.shape[-2], x2.shape[-1]
        xf = x2.view(-1, D_)
        y = torch.empty_like(xf)
        BLOCK = triton.next_power_of_2(D_)
        _rmsnorm_gelu_kernel[(xf.shape[0],)](
            xf, self.rms0_w, y,
            xf.stride(0), y.stride(0),
            D_=D_, BLOCK=BLOCK,
            num_warps=8,
        )
        return y.view(x2.shape)

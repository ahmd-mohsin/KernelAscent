import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 436
M, D, DT = 4096, 4096, torch.float16


@triton.jit
def _fused_scale_rmsnorm_kernel(
    X_ptr, W_ptr, Y_ptr,
    N,
    stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # load input row (fp16) -> fp32
    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # x = x * 1.1783 : PyTorch computes half elementwise ops in fp32 (opmath)
    # then rounds result back to fp16
    x = (x * 1.1783).to(tl.float16)

    # _xf = x.float()
    xf = x.to(tl.float32)

    # mean of squares + rsqrt (matches torch.rsqrt -> CUDA rsqrtf)
    ms = tl.sum(xf * xf, axis=0) / N
    r = tl.math.rsqrt(ms + 1e-6)

    # (xf * r) rounded to fp16, then multiplied by rms weight
    # (fp16 * fp16 done in fp32 opmath, rounded to fp16)
    xn = (xf * r).to(tl.float16).to(tl.float32)
    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (xn * w).to(tl.float16)

    tl.store(Y_ptr + row * stride_y + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.W2 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _fused_scale_rmsnorm_kernel[(m,)](
            x, self.rms1_w, y,
            n,
            x.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y @ self.W2

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 619
M, D, DT = 8192, 512, torch.bfloat16


@triton.jit
def _softmax_scale_bias_kernel(
    X, B, Y,
    stride_xm, stride_ym,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=-float('inf'))
    x = x.to(tl.float32)

    # softmax in fp32 (matches PyTorch's internal fp32 accumulation for bf16)
    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = e / s

    # round to bf16 (softmax output), then scale in fp32 and round again
    sm_bf16 = sm.to(tl.bfloat16)
    scaled = (sm_bf16.to(tl.float32) * 1.0686).to(tl.bfloat16)

    # add bias in fp32, round to bf16
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    out = (scaled.to(tl.float32) + b).to(tl.bfloat16)

    tl.store(Y + row * stride_ym + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b2 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 1024 else 8
        _softmax_scale_bias_kernel[(rows,)](
            x2, self.b2, y,
            x2.stride(0), y.stride(0),
            N=N, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)

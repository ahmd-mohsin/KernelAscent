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
    X, B, Out,
    n_cols,
    stride_xm, stride_om,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=float('-inf')).to(tl.float32)

    row_max = tl.max(x, axis=0)
    x = x - row_max
    num = tl.exp(x)
    denom = tl.sum(num, axis=0)
    sm = num / denom

    # match PyTorch: softmax output rounded to bf16 first
    sm_bf = sm.to(tl.bfloat16)
    # scale in fp32, round to bf16
    y = (sm_bf.to(tl.float32) * SCALE).to(tl.bfloat16)
    # add bias in fp32, round to bf16
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    z = (y.to(tl.float32) + b).to(tl.bfloat16)

    tl.store(Out + row * stride_om + cols, z, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b2 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = torch.softmax(x, dim=-1)
            y = y * 1.0686
            return y + self.b2

        x = x.contiguous()
        M_, D_ = x.shape[-2] if x.dim() > 1 else 1, x.shape[-1]
        orig_shape = x.shape
        x2 = x.view(-1, D_)
        n_rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(D_)
        num_warps = 4 if BLOCK <= 1024 else 8

        _softmax_scale_bias_kernel[(n_rows,)](
            x2, self.b2, out,
            D_,
            x2.stride(0), out.stride(0),
            SCALE=1.0686,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)

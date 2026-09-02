import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 778
M, D, DT = 512, 2049, torch.bfloat16


@triton.jit
def _fused_relu_bias_scale_softmax(
    X_ptr, B_ptr, Y_ptr,
    n_cols,
    stride_x, stride_y,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # relu (exact in bf16)
    x = tl.maximum(x, 0.0)
    # add bias: fp32 compute, round to bf16 (matches PyTorch opmath semantics)
    v = (x + b).to(tl.bfloat16).to(tl.float32)
    # scale: fp32 compute, round to bf16
    v = (v * SCALE).to(tl.bfloat16).to(tl.float32)

    # softmax in fp32 (matches PyTorch's internal float accumulation)
    v = tl.where(mask, v, float('-inf'))
    row_max = tl.max(v, axis=0)
    e = tl.exp(v - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    y = e / denom

    tl.store(Y_ptr + row * stride_y + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = torch.relu(x)
            x = x + self.b1
            x = x * 1.4024
            return torch.softmax(x, dim=-1)

        orig_shape = x.shape
        n_cols = orig_shape[-1]
        x2d = x.contiguous().view(-1, n_cols)
        n_rows = x2d.shape[0]

        b = self.b1
        if b.device != x.device:
            b = b.to(x.device)

        y = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16

        _fused_relu_bias_scale_softmax[(n_rows,)](
            x2d, b, y,
            n_cols,
            x2d.stride(0), y.stride(0),
            SCALE=1.4024,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)

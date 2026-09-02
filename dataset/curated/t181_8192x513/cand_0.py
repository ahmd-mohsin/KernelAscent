import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 181
M, D, DT = 8192, 513, torch.bfloat16


@triton.jit
def _fused_scale_bias_softmax(
    X, B, Y,
    n_cols,
    stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)

    # Emulate bf16 elementwise ops (fp32 compute + round back to bf16,
    # which matches hardware bf16 arithmetic semantics)
    xf = x.to(tl.float32)
    xf = (xf * 1.2248).to(tl.bfloat16).to(tl.float32)
    xf = (xf * 1.0339).to(tl.bfloat16).to(tl.float32)
    xf = (xf + b.to(tl.float32)).to(tl.bfloat16).to(tl.float32)

    # Softmax with fp32 accumulation (matches PyTorch's accscalar behavior)
    xf = tl.where(mask, xf, float('-inf'))
    row_max = tl.max(xf, axis=0)
    num = tl.exp(xf - row_max)
    num = tl.where(mask, num, 0.0)
    denom = tl.sum(num, axis=0)
    out = num / denom

    tl.store(Y + row * stride_y + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b2 = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = x * 1.2248
            y = y * 1.0339
            y = y + self.b2
            return torch.softmax(y, dim=-1)

        x = x.contiguous()
        orig_shape = x.shape
        n_cols = orig_shape[-1]
        x2d = x.view(-1, n_cols)
        n_rows = x2d.shape[0]

        b = self.b2
        if b.device != x.device:
            b = b.to(x.device)

        y = torch.empty_like(x2d)
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 4 if BLOCK <= 1024 else 8

        _fused_scale_bias_softmax[(n_rows,)](
            x2d, b, y,
            n_cols,
            x2d.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)

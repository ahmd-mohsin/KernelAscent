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
    stride_xm, stride_ym,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)

    # Match PyTorch semantics: scalar mul on bf16 uses fp32 opmath, rounds back to bf16
    x1 = (x.to(tl.float32) * 1.2248).to(tl.bfloat16)
    x2 = (x1.to(tl.float32) * 1.0339).to(tl.bfloat16)
    x3 = (x2.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)

    # Softmax computed in fp32 (PyTorch upcasts bf16 to float internally)
    v = tl.where(mask, x3.to(tl.float32), float('-inf'))
    row_max = tl.max(v, axis=0)
    num = tl.exp(v - row_max)
    num = tl.where(mask, num, 0.0)
    denom = tl.sum(num, axis=0)
    out = (num / denom).to(tl.bfloat16)

    tl.store(Y + row * stride_ym + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b2 = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        n_cols = orig_shape[-1]
        x2d = x.reshape(-1, n_cols)
        if not x2d.is_contiguous():
            x2d = x2d.contiguous()
        n_rows = x2d.shape[0]
        y = torch.empty_like(x2d)
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_scale_bias_softmax[(n_rows,)](
            x2d, self.b2, y,
            n_cols,
            x2d.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.reshape(orig_shape)

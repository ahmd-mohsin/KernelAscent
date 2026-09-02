import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 578
M, D, DT = 2048, 1024, torch.bfloat16


@triton.jit
def _fused_softmax_kernel(
    x_ptr, b0_ptr, b3_ptr, out_ptr,
    n_cols,
    stride_x, stride_out,
    scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0)
    b0 = tl.load(b0_ptr + cols, mask=mask, other=0.0)
    b3 = tl.load(b3_ptr + cols, mask=mask, other=0.0)

    # x + b0 in fp32, round to bf16 (matches PyTorch bf16 elementwise semantics)
    t = (x.to(tl.float32) + b0.to(tl.float32)).to(tl.bfloat16)
    # scale in fp32, round to bf16
    t = (t.to(tl.float32) * scale).to(tl.bfloat16)

    # softmax in fp32 (PyTorch upcasts bf16 softmax internally)
    tf = t.to(tl.float32)
    tf = tl.where(mask, tf, float('-inf'))
    m = tl.max(tf, axis=0)
    e = tl.exp(tf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = (e / s).to(tl.bfloat16)

    out = (sm.to(tl.float32) + b3.to(tl.float32)).to(tl.bfloat16)
    tl.store(out_ptr + row * stride_out + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = x + self.b0
            y = y * 1.4701
            y = torch.softmax(y, dim=-1)
            return y + self.b3

        orig_shape = x.shape
        x2 = x.contiguous().view(-1, orig_shape[-1])
        n_rows, n_cols = x2.shape
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 4 if BLOCK <= 1024 else 8

        _fused_softmax_kernel[(n_rows,)](
            x2, self.b0, self.b3, out,
            n_cols,
            x2.stride(0), out.stride(0),
            1.4701,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)

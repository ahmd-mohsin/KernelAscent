"""Sample candidate: single-block Triton row-softmax.

Fast for moderate row widths, and it deliberately fails on very wide rows
(BLOCK exceeds what fits), which the harness records as a correctness failure.
"""
import torch, triton
import triton.language as tl

@triton.jit
def _softmax_kernel(inp_ptr, out_ptr, in_stride, out_stride, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(inp_ptr + row * in_stride + cols, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    tl.store(out_ptr + row * out_stride + cols, num / den, mask=mask)

def run(x):
    x = x.contiguous()
    n_rows, n_cols = x.shape
    BLOCK = triton.next_power_of_2(n_cols)
    nw = 4 if BLOCK <= 2048 else (8 if BLOCK <= 8192 else 16)
    out = torch.empty_like(x)
    _softmax_kernel[(n_rows,)](x, out, x.stride(0), out.stride(0), n_cols, BLOCK=BLOCK, num_warps=nw)
    return out

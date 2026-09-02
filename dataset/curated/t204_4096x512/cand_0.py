import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 204
M, D, DT = 4096, 512, torch.float16


@triton.jit
def _fused_act_softmax_kernel(
    X_ptr, Y_ptr,
    N,
    stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # relu
    x = tl.maximum(x, 0.0)

    # exact (erf-based) gelu, computed in fp32 then rounded to fp16 (matches PyTorch)
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.float16).to(tl.float32)

    # scale, rounded to fp16 (matches PyTorch elementwise output dtype)
    v = g * 1.2304
    v = v.to(tl.float16).to(tl.float32)

    # softmax with fp32 accumulation (matches PyTorch half softmax)
    v = tl.where(mask, v, float('-inf'))
    m = tl.max(v, axis=0)
    e = tl.exp(v - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y_ptr + row * stride_y + offs, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores on A100)
        h = x @ self.W0

        orig_shape = h.shape
        h2 = h.reshape(-1, orig_shape[-1])
        if not h2.is_contiguous():
            h2 = h2.contiguous()

        rows, N = h2.shape
        y = torch.empty_like(h2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_act_softmax_kernel[(rows,)](
            h2, y,
            N,
            h2.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )

        return y.reshape(orig_shape)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 156
M, D, DT = 4096, 4096, torch.float16


@triton.jit
def _fused_scale_relu_bias_softmax(
    X, B3, B4, Y,
    N, stride_x, stride_y,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(B4 + cols, mask=mask, other=0.0).to(tl.float32)

    # x * 1.3741 : fp16 op done in fp32 opmath, rounded to fp16 (matches PyTorch)
    t = (x * SCALE).to(tl.float16).to(tl.float32)
    # relu (exact in any precision)
    t = tl.maximum(t, 0.0)
    # + b3 (round to fp16 like PyTorch fp16 add)
    t = (t + b3).to(tl.float16).to(tl.float32)
    # + b4
    t = (t + b4).to(tl.float16).to(tl.float32)

    # softmax in fp32 accumulation (matches PyTorch fp16 softmax internals)
    t = tl.where(mask, t, float('-inf'))
    m = tl.max(t, axis=0)
    e = tl.exp(t - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = (e / s).to(tl.float16)

    tl.store(Y + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores) - identical op to reference
        y = x @ self.W0

        orig_shape = y.shape
        N = orig_shape[-1]
        y2 = y.reshape(-1, N)
        if not y2.is_contiguous():
            y2 = y2.contiguous()
        rows = y2.shape[0]

        out = torch.empty_like(y2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 4
        if BLOCK >= 1024:
            num_warps = 8
        if BLOCK >= 4096:
            num_warps = 16

        _fused_scale_relu_bias_softmax[(rows,)](
            y2, self.b3, self.b4, out,
            N, y2.stride(0), out.stride(0),
            SCALE=1.3741,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.reshape(orig_shape)

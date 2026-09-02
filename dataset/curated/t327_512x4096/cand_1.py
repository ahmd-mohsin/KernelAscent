import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 327
M, D, DT = 512, 4096, torch.float16


@triton.jit
def _bias_scale_bias_softmax_kernel(
    X_ptr, B1_ptr, B3_ptr, OUT_ptr,
    N, SCALE,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # Match PyTorch elementwise semantics on fp16 tensors:
    # each op computed in fp32 (opmath) then rounded back to fp16.
    y = (x + b1).to(tl.float16).to(tl.float32)
    y = (y * SCALE).to(tl.float16).to(tl.float32)
    y = (y + b3).to(tl.float16).to(tl.float32)

    # Softmax in fp32 accumulation (matches PyTorch half softmax accum type)
    y = tl.where(mask, y, float("-inf"))
    m = tl.max(y, 0)
    e = tl.exp(y - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, 0)
    out = (e / s).to(tl.float16)
    tl.store(OUT_ptr + row * N + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 4096, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS fp16 tensor-core GEMM (fp32 accumulate) — already optimal on A100
        y = x @ self.W0

        if not y.is_cuda:
            y = y + self.b1
            y = y * 1.1132
            y = y + self.b3
            return torch.softmax(y, dim=-1)

        y = y.contiguous()
        rows, N = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _bias_scale_bias_softmax_kernel[(rows,)](
            y, self.b1, self.b3, out,
            N, 1.1132,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out

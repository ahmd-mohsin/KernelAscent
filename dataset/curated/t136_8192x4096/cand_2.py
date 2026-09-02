import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 136
M, D, DT = 8192, 4096, torch.bfloat16


@triton.jit
def _fused_softmax_gelu_kernel(
    X, B, Y,
    N,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)

    # x * 1.0831 (bf16 rounding, matching eager elementwise semantics)
    t = (x * 1.0831).to(tl.bfloat16).to(tl.float32)
    # * 1.4976
    t = (t * 1.4976).to(tl.bfloat16).to(tl.float32)
    # + b2 (opmath fp32, rounded back to bf16)
    t = (t + b).to(tl.bfloat16).to(tl.float32)

    # softmax in fp32 accumulation (as PyTorch does for bf16)
    t = tl.where(mask, t, float('-inf'))
    m = tl.max(t, axis=0)
    e = tl.exp(t - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = (e / s).to(tl.bfloat16).to(tl.float32)

    # exact gelu: y * 0.5 * (1 + erf(y / sqrt(2))) computed in fp32
    g = y * 0.5 * (1.0 + tl.math.erf(y * 0.7071067811865476))

    tl.store(Y + row * N + offs, g.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b2 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            t = x * 1.0831
            t = t * 1.4976
            t = t + self.b2
            t = torch.softmax(t, dim=-1)
            return F.gelu(t)

        orig_shape = x.shape
        x2 = x.contiguous().view(-1, orig_shape[-1])
        rows, N = x2.shape
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_softmax_gelu_kernel[(rows,)](
            x2, self.b2, y, N,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)

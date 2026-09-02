import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 725
M, D, DT = 512, 4097, torch.bfloat16

INV_SQRT2 = 0.7071067811865476


@triton.jit
def _fused_gelu2_bias_softmax(
    X, B, Y,
    N,
    stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # gelu #1 (exact erf gelu, computed in fp32, rounded back to bf16 like PyTorch)
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # gelu #2
    g = 0.5 * g * (1.0 + tl.math.erf(g * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # bias add (bf16 add semantics: fp32 compute, round to bf16)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    s = (g + b).to(tl.bfloat16).to(tl.float32)

    # softmax in fp32 (matches PyTorch bf16 softmax internal fp32 accumulation)
    s = tl.where(mask, s, float('-inf'))
    m = tl.max(s, axis=0)
    e = tl.exp(s - m)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    y = e / denom

    tl.store(Y + row * stride_y + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b2 = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = F.gelu(x)
            y = F.gelu(y)
            y = y + self.b2
            return torch.softmax(y, dim=-1)

        orig_shape = x.shape
        n = orig_shape[-1]
        x2d = x.contiguous().view(-1, n)
        rows = x2d.shape[0]

        out = torch.empty_like(x2d)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 4096 else 4

        _fused_gelu2_bias_softmax[(rows,)](
            x2d, self.b2, out,
            n,
            x2d.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 971
M, D, DT = 2048, 1024, torch.bfloat16


@triton.jit
def _fused_gelu_relu_bias_softmax(
    Y_ptr, B_ptr, O_ptr,
    N, stride_y, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    y = tl.load(Y_ptr + row * stride_y + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # GELU (erf-based) computed in fp32, rounded to bf16 (matches PyTorch bf16 gelu)
    g = 0.5 * y * (1.0 + tl.math.erf(y * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # ReLU
    r = tl.maximum(g, 0.0)

    # bias add in bf16 precision (fp32 add of exactly-representable values, round to bf16)
    s = (r + b).to(tl.bfloat16).to(tl.float32)

    # softmax in fp32 (matches PyTorch bf16 softmax acc type)
    s = tl.where(mask, s, float('-inf'))
    m = tl.max(s, axis=0)
    e = tl.exp(s - m)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(O_ptr + row * stride_o + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 1024, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0  # cuBLAS bf16 matmul (tensor cores)
        m, n = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(n)
        _fused_gelu_relu_bias_softmax[(m,)](
            y, self.b3, out,
            n, y.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out

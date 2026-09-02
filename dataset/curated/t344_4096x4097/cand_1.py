import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 344
M, D, DT = 4096, 4097, torch.bfloat16


@triton.jit
def _fused_gelu_scale_bias_softmax(
    X_ptr, B_ptr, Out_ptr,
    N, stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # exact GELU (erf variant), computed in fp32 then rounded to bf16 (matches PyTorch opmath)
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # scalar multiplies: fp32 compute, round to bf16 each step (matches PyTorch semantics)
    g = (g * 1.0846).to(tl.bfloat16).to(tl.float32)
    g = (g * 1.379).to(tl.bfloat16).to(tl.float32)

    # bias add in fp32, round to bf16
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    g = (g + b).to(tl.bfloat16).to(tl.float32)

    # softmax in fp32
    g = tl.where(mask, g, float('-inf'))
    m = tl.max(g, axis=0)
    e = tl.exp(g - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.bfloat16)

    tl.store(Out_ptr + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 2048, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (bf16 with fp32 accumulate, same as reference)
        y = x @ self.W0
        y = y.contiguous()

        Mrows, N = y.shape
        out = torch.empty_like(y)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_gelu_scale_bias_softmax[(Mrows,)](
            y, self.b4, out,
            N, y.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out

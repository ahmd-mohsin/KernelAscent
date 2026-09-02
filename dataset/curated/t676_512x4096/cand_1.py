import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 676
M, D, DT = 512, 4096, torch.bfloat16


@triton.jit
def _fused_relu_gelu_bias_softmax(
    X_ptr, B_ptr, Out_ptr,
    N, stride_xm, stride_om,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0)  # bf16
    # relu in bf16 (exact)
    r = tl.maximum(x, 0.0)
    # gelu computed in fp32 then rounded to bf16 (matches PyTorch opmath)
    rf = r.to(tl.float32)
    g = rf * 0.5 * (1.0 + tl.math.erf(rf * 0.7071067811865476))
    g_bf = g.to(tl.bfloat16)
    # bias add: fp32 compute, round to bf16 (matches PyTorch binary op)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0)  # bf16
    y = (g_bf.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)
    # softmax in fp32 on the bf16 values (matches PyTorch softmax accumulate)
    s = y.to(tl.float32)
    s = tl.where(mask, s, float('-inf'))
    m = tl.max(s, axis=0)
    e = tl.exp(s - m)
    denom = tl.sum(e, axis=0)
    out = (e / denom).to(tl.bfloat16)
    tl.store(Out_ptr + row * stride_om + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 2048, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 GEMM with fp32 accumulate
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(n)
        _fused_relu_gelu_bias_softmax[(m,)](
            h, self.b3, out,
            n, h.stride(0), out.stride(0),
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return out

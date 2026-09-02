import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 664
M, D, DT = 8192, 4096, torch.float16


@triton.jit
def _fused_epilogue_kernel(
    Y_ptr, B_ptr, W_ptr, Out_ptr,
    N, stride_y, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    # load matmul output row (fp16 -> fp32 compute, like PyTorch opmath)
    y = tl.load(Y_ptr + row * stride_y + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # bias add (fp32 compute, round to fp16 like separate op)
    x = (y + b).to(tl.float16).to(tl.float32)

    # exact GELU: 0.5 * x * (1 + erf(x / sqrt(2))) in fp32, round to fp16
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.float16).to(tl.float32)

    # softmax (fp32 accumulate, fp16 output like PyTorch half softmax)
    g_masked = tl.where(mask, g, float('-inf'))
    m = tl.max(g_masked, axis=0)
    e = tl.exp(g_masked - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = (e / s).to(tl.float16).to(tl.float32)

    # RMSNorm in fp32, cast to fp16, then multiply by weight (fp32 compute -> fp16)
    ms = tl.sum(sm * sm, axis=0) / N
    r = (sm * tl.math.rsqrt(ms + 1e-6)).to(tl.float16).to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = (r * w).to(tl.float16)

    # ReLU
    out = tl.maximum(out, 0.0)

    tl.store(Out_ptr + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    @torch.no_grad()
    def forward(self, x):
        # matmul via cuBLAS tensor cores
        y = x @ self.W0
        y = y.contiguous()
        Mrows, N = y.shape
        out = torch.empty_like(y)

        BLOCK = triton.next_power_of_2(N)
        _fused_epilogue_kernel[(Mrows,)](
            y, self.b1, self.rms4_w, out,
            N, y.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out

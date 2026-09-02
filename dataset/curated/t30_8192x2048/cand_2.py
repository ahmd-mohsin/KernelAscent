import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 30
M, D, DT = 8192, 2048, torch.float16


@triton.jit
def _fused_bias_softmax_rms_kernel(
    X_ptr, B_ptr, W_ptr, OUT_ptr,
    N,                      # row length (4096)
    scale,                  # 1.0129
    eps,                    # 1e-6
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # load matmul output (fp16) and bias (fp16)
    x = tl.load(X_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # x + b1 : elementwise add in fp32 opmath, round to fp16 (matches torch)
    v = (x + b).to(tl.float16).to(tl.float32)
    v = tl.where(mask, v, float('-inf'))

    # softmax in fp32, cast result to fp16 (matches torch half softmax)
    mx = tl.max(v, axis=0)
    e = tl.exp(v - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = (e / s).to(tl.float16)

    # x * 1.0129 : fp32 opmath, round to fp16
    y = (sm.to(tl.float32) * scale).to(tl.float16)

    # RMS norm in fp32
    xf = y.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    r = 1.0 / tl.sqrt(ms + eps)
    normed = (xf * r).to(tl.float16)

    # multiply by rms weight (fp16 elementwise, fp32 opmath)
    w = tl.load(W_ptr + offs, mask=mask, other=0.0)
    out = (normed.to(tl.float32) * w.to(tl.float32)).to(tl.float16)

    tl.store(OUT_ptr + row * N + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS tensor cores (fp16 in, fp32 accumulate) - same as reference
        y = x @ self.W0
        y = y.contiguous()

        Mrows, N = y.shape
        out = torch.empty_like(y)

        BLOCK = triton.next_power_of_2(N)
        _fused_bias_softmax_rms_kernel[(Mrows,)](
            y, self.b1, self.rms4_w, out,
            N, 1.0129, 1e-6,
            BLOCK=BLOCK,
            num_warps=16,
        )
        return out

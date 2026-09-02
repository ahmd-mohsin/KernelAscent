import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 386
M, D, DT = 8192, 2048, torch.bfloat16


@triton.jit
def _fused_rms_softmax_gelu_kernel(
    X_ptr, W_ptr, Y_ptr,
    stride_x, stride_y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    # load row (bf16 output of matmul), upcast to fp32
    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm in fp32, then cast to bf16 (matches (_xf * rsqrt(...)).to(bf16))
    ms = tl.sum(x * x, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    xn = (x * inv).to(tl.bfloat16)

    # multiply by weight: bf16*bf16 elementwise -> fp32 opmath -> bf16 result
    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    y = (xn.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    # softmax: upcast to fp32, compute, downcast to bf16 (matches torch bf16 softmax)
    yf = y.to(tl.float32)
    yf = tl.where(mask, yf, float('-inf'))
    m = tl.max(yf, axis=0)
    e = tl.exp(yf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = (e / s).to(tl.bfloat16)

    # scale by 1.4014 (fp32 opmath -> bf16)
    z = (sm.to(tl.float32) * 1.4014).to(tl.bfloat16)

    # gelu (erf-based, fp32 opmath -> bf16)
    zf = z.to(tl.float32)
    g = 0.5 * zf * (1.0 + tl.math.erf(zf * 0.7071067811865476))
    gb = g.to(tl.bfloat16)

    # relu on bf16 values
    out = tl.maximum(gb.to(tl.float32), 0.0).to(tl.bfloat16)

    tl.store(Y_ptr + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS (tensor cores)
        h = x @ self.W0  # (M, 512) bf16
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        _fused_rms_softmax_gelu_kernel[(Mrows,)](
            h, self.rms1_w, out,
            h.stride(0), out.stride(0),
            N=N,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out

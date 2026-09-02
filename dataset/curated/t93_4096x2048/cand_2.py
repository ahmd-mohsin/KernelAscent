import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 93
M, D, DT = 4096, 2048, torch.float16


@triton.jit
def _fused_relu_softmax_bias_rms_kernel(
    X_ptr, B_ptr, W_ptr, Out_ptr,
    N, stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # ReLU
    x = tl.where(mask, tl.maximum(x, 0.0), float('-inf'))

    # Softmax (fp32 accumulation, matching PyTorch half softmax)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = (e / s).to(tl.float16)

    # Bias add in fp16 (matching reference dtype semantics)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    y = p + b

    # RMSNorm in fp32
    yf = y.to(tl.float32)
    ms = tl.sum(tl.where(mask, yf * yf, 0.0), axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    z = (yf * inv).to(tl.float16)

    # Scale by weight in fp16
    w = tl.load(W_ptr + offs, mask=mask, other=0.0)
    out = z * w

    tl.store(Out_ptr + row * stride_o + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores)
        h = x @ self.W0

        rows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)

        _fused_relu_softmax_bias_rms_kernel[(rows,)](
            h, self.b3, self.rms4_w, out,
            N, h.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=16,
            num_stages=1,
        )
        return out

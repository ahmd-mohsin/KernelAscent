import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 93
M, D, DT = 4096, 2048, torch.float16


@triton.jit
def _fused_relu_softmax_bias_rms(
    X_ptr, B_ptr, W_ptr, Out_ptr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    x_ptrs = X_ptr + row * N + offs

    # load row, upcast to fp32 (matches PyTorch softmax half->float accumulation)
    x = tl.load(x_ptrs).to(tl.float32)

    # relu
    x = tl.maximum(x, 0.0)

    # softmax in fp32
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = (e / s).to(tl.float16)

    # bias add in fp16 (matches reference dtype behavior)
    b = tl.load(B_ptr + offs)
    y = y + b

    # RMSNorm: fp32 mean of squares, rsqrt, cast back, scale in fp16
    yf = y.to(tl.float32)
    ms = tl.sum(yf * yf, axis=0) / N
    r = tl.math.rsqrt(ms + 1e-6)
    w = tl.load(W_ptr + offs)
    out = (yf * r).to(tl.float16) * w

    tl.store(Out_ptr + row * N + offs, out)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores on A100)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        M_, N_ = h.shape
        out = torch.empty_like(h)
        _fused_relu_softmax_bias_rms[(M_,)](
            h, self.b3, self.rms4_w, out,
            N=N_, BLOCK=N_,
            num_warps=16,
        )
        return out

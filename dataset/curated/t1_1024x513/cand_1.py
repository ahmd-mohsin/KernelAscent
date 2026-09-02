import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 1
M, D, DT = 1024, 513, torch.bfloat16


@triton.jit
def _fused_rms_softmax_kernel(
    X_ptr, B1_ptr, W_ptr, B4_ptr, Out_ptr,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)

    # load row of matmul output (bf16) and bias
    x = tl.load(X_ptr + row * N + offs)
    b1 = tl.load(B1_ptr + offs)

    # x = x + b1  (bf16 add: fp32 compute, round to bf16)
    xb = (x.to(tl.float32) + b1.to(tl.float32)).to(tl.bfloat16)

    # RMSNorm in fp32
    xf = xb.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    xn = (xf * inv).to(tl.bfloat16)  # cast to bf16 (matches .to(x.dtype))

    # * rms2_w  (bf16 mul: fp32 compute, round to bf16)
    w = tl.load(W_ptr + offs)
    y = (xn.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    # softmax (fp32 accumulate, bf16 output)
    yf = y.to(tl.float32)
    mx = tl.max(yf, axis=0)
    e = tl.exp(yf - mx)
    s = tl.sum(e, axis=0)
    sm = (e / s).to(tl.bfloat16)

    # + b4 then relu (bf16 add: fp32 compute, round to bf16)
    b4 = tl.load(B4_ptr + offs)
    z = (sm.to(tl.float32) + b4.to(tl.float32)).to(tl.bfloat16)
    z = tl.maximum(z, tl.zeros_like(z))

    tl.store(Out_ptr + row * N + offs, z)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 4096, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        _fused_rms_softmax_kernel[(m,)](
            h, self.b1, self.rms2_w, self.b4, out,
            N=n, BLOCK=n,
            num_warps=16,
        )
        return out

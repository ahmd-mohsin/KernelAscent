import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 429
M, D, DT = 8192, 512, torch.bfloat16


@triton.jit
def _rmsnorm_softmax_kernel(
    X, W, Out,
    N, stride_x, stride_o,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # RMSNorm (mean over N, computed in fp32)
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + eps)
    xn = xf * inv
    # round to bf16 as in reference (.to(x.dtype))
    xn_bf = xn.to(tl.bfloat16)

    w = tl.load(W + cols, mask=mask, other=0.0)
    # bf16 * bf16 -> fp32 compute, round to bf16 (PyTorch semantics)
    z = (xn_bf.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    # softmax in fp32 (PyTorch upcasts bf16 softmax internally)
    zf = z.to(tl.float32)
    zf = tl.where(mask, zf, float('-inf'))
    zmax = tl.max(zf, axis=0)
    e = tl.exp(zf - zmax)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(Out + row * stride_o + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS bf16 GEMM
        m, n = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _rmsnorm_softmax_kernel[(m,)](
            x, self.rms1_w, out,
            n, x.stride(0), out.stride(0),
            1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out

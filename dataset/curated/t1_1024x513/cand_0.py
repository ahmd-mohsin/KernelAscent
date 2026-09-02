import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 1
M, D, DT = 1024, 513, torch.bfloat16


@triton.jit
def _fused_bias_rms_softmax_bias_relu(
    X, B1, W, B4, Out,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    ptr = X + row * N + offs

    # bias add (bf16 elementwise op: fp32 opmath, round back to bf16)
    x = tl.load(ptr).to(tl.float32)
    b1 = tl.load(B1 + offs).to(tl.float32)
    x = (x + b1).to(tl.bfloat16).to(tl.float32)

    # RMSNorm in fp32, round to bf16, then scale by weight (bf16 op, fp32 opmath)
    ms = tl.sum(x * x, axis=0) / N
    r = tl.rsqrt(ms + 1e-6)
    xn = (x * r).to(tl.bfloat16).to(tl.float32)
    w = tl.load(W + offs).to(tl.float32)
    y = (xn * w).to(tl.bfloat16).to(tl.float32)

    # softmax (fp32 accumulation, bf16 output)
    m = tl.max(y, axis=0)
    e = tl.exp(y - m)
    s = tl.sum(e, axis=0)
    sm = (e / s).to(tl.bfloat16).to(tl.float32)

    # bias add + relu
    b4 = tl.load(B4 + offs).to(tl.float32)
    o = sm + b4
    o = tl.where(o > 0.0, o, 0.0)
    tl.store(Out + row * N + offs, o.to(tl.bfloat16))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 4096, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores, bf16)
        x = x @ self.W0
        x = x.contiguous()
        rows, N = x.shape
        out = torch.empty_like(x)
        _fused_bias_rms_softmax_bias_relu[(rows,)](
            x, self.b1, self.rms2_w, self.b4, out,
            N=N, BLOCK=N,
            num_warps=8,
        )
        return out


def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(M, D, generator=g).to(DT)]

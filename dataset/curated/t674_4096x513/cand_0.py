import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 674
M, D, DT = 4096, 513, torch.float16


@triton.jit
def _fused_rms_ln_relu(X, RW, G, B, Y, stride,
                       N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride + offs, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm (stats in fp32, downcast to fp16, then fp16 multiply with weight)
    ms = tl.sum(x * x, axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    xh = (x * r).to(tl.float16)
    rw = tl.load(RW + offs, mask=mask, other=0.0)
    v = xh * rw  # fp16 multiply (matches reference semantics)

    # LayerNorm in fp32 (matches PyTorch half layer_norm which accumulates in fp32)
    vf = v.to(tl.float32)
    mean = tl.sum(vf, axis=0) / N
    d = tl.where(mask, vf - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    inv = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    y = (vf - mean) * inv * g + b

    # ReLU
    y = tl.maximum(y, 0.0)

    tl.store(Y + row * stride + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 512, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS tensor cores
        x = x @ self.W0
        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        _fused_rms_ln_relu[(m,)](
            x, self.rms1_w, self.ln2_g, self.ln2_b, y,
            x.stride(0), N=n, BLOCK=triton.next_power_of_2(n),
            num_warps=4,
        )
        return y

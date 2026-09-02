import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 393
M, D, DT = 2048, 2049, torch.float16


@triton.jit
def _gelu_bias_gelu_softmax(X, B, Y, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * N + offs, mask=mask, other=0.0)

    # GELU (exact, fp32 opmath, round to fp16 like PyTorch)
    xf = x.to(tl.float32)
    g = (xf * 0.5 * (1.0 + tl.math.erf(xf * 0.7071067811865476))).to(tl.float16)

    # bias add in fp16 (matches half+half elementwise add)
    b = tl.load(B + offs, mask=mask, other=0.0)
    h = g + b

    # second GELU
    hf = h.to(tl.float32)
    t = (hf * 0.5 * (1.0 + tl.math.erf(hf * 0.7071067811865476))).to(tl.float16)

    # softmax in fp32 accumulation (matches PyTorch half softmax)
    tf = tl.where(mask, t.to(tl.float32), float('-inf'))
    mx = tl.max(tf, 0)
    e = tl.exp(tf - mx)
    s = tl.sum(e, 0)
    out = (e / s).to(tl.float16)

    tl.store(Y + row * N + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 1024, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.W5 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)

    def forward(self, x):
        # first matmul via cuBLAS tensor cores
        h = torch.matmul(x, self.W0)
        h = h.contiguous()

        m, n = h.shape
        BLOCK = triton.next_power_of_2(n)
        # fused: gelu -> +bias -> gelu -> softmax (in-place on h)
        _gelu_bias_gelu_softmax[(m,)](
            h, self.b2, h, n,
            BLOCK=BLOCK,
            num_warps=8,
        )

        # second matmul via cuBLAS tensor cores
        return torch.matmul(h, self.W5)

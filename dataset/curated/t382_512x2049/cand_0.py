import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 382
M, D, DT = 512, 2049, torch.float16


@triton.jit
def _softmax_layernorm_kernel(
    X, Y, G, B,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    ptr = X + row * N + offs

    # load row in fp32
    x = tl.load(ptr).to(tl.float32)

    # softmax (fp32 accumulation, like torch)
    m = tl.max(x, 0)
    e = tl.exp(x - m)
    z = tl.sum(e, 0)
    s = e / z

    # round to fp16 (softmax output dtype), then read back for layernorm stats
    s = s.to(tl.float16).to(tl.float32)

    # layernorm stats in fp32
    mean = tl.sum(s, 0) / N
    d = s - mean
    var = tl.sum(d * d, 0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + offs).to(tl.float32)
    b = tl.load(B + offs).to(tl.float32)
    y = d * rstd * g + b

    tl.store(Y + row * N + offs, y.to(tl.float16))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 4096, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS tensor cores
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        _softmax_layernorm_kernel[(Mrows,)](
            h, out, self.ln2_g, self.ln2_b,
            N, 1e-5,
            BLOCK=4096,
            num_warps=8,
        )
        return out

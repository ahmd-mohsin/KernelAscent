import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 568
M, D, DT = 512, 512, torch.bfloat16


@triton.jit
def _fused_epilogue(X, Wg, B3, B4, Out,
                    D: tl.constexpr, EPS: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, D)
    ptr = row * D + offs

    x = tl.load(X + ptr).to(tl.float32)

    # exact (erf) GELU in fp32
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    # match reference: gelu output rounded to bf16 before rmsnorm
    g = g.to(tl.bfloat16).to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(g * g, axis=0) / D
    n = g * tl.math.rsqrt(ms + EPS)

    # back to bf16, scale and biases (sequential, matching reference)
    y = n.to(tl.bfloat16)
    y = y * tl.load(Wg + offs)
    y = y + tl.load(B3 + offs)
    y = y + tl.load(B4 + offs)

    tl.store(Out + ptr, y)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS (tensor cores)
        h = torch.matmul(x, self.W0)
        h2 = h.reshape(-1, 512).contiguous()
        n_rows = h2.shape[0]
        out = torch.empty_like(h2)
        _fused_epilogue[(n_rows,)](
            h2, self.rms2_w, self.b3, self.b4, out,
            D=512, EPS=1e-6,
            num_warps=4,
        )
        return out.reshape(h.shape)

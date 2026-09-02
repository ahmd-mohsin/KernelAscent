import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 489
M, D, DT = 1024, 4096, torch.float16


@triton.jit
def _ln_kernel(X, G, B, Y, D: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    x = tl.load(X + row * D + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / D
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + EPS)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x - mean) * rstd * g + b
    tl.store(Y + row * D + cols, y.to(tl.float16), mask=mask)


@triton.jit
def _epilogue_kernel(Z, B, Y, N, C, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    z = tl.load(Z + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + (offs % C), mask=mask, other=0.0).to(tl.float32)
    z = tl.maximum(z, 0.0)
    y = tl.maximum(z + b, 0.0)
    tl.store(Y + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.W1 = nn.Parameter((torch.randn(4096, 2048, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        rows, d = x.shape

        # Fused LayerNorm (fp32 accumulation, matching F.layer_norm semantics)
        xn = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        _ln_kernel[(rows,)](
            x, self.ln0_g, self.ln0_b, xn,
            D=d, EPS=1e-5, BLOCK=BLOCK,
            num_warps=8,
        )

        # Tensor-core GEMM via cuBLAS (fp16 with fp32 accumulate, same as reference)
        z = xn @ self.W1

        # Fused epilogue: relu -> +bias -> relu
        out = torch.empty_like(z)
        n = z.numel()
        c = z.shape[-1]
        EBLOCK = 1024
        _epilogue_kernel[(triton.cdiv(n, EBLOCK),)](
            z, self.b3, out, n, c,
            BLOCK=EBLOCK, num_warps=4,
        )
        return out

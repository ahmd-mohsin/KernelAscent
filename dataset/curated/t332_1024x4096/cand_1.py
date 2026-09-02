import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 332
M, D, DT = 1024, 4096, torch.bfloat16


@triton.jit
def _fused_scale_relu_gelu_softmax(
    X, Y,
    N,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    ptr = X + row * N + offs

    x = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)

    # scale (rounded to bf16 as PyTorch elementwise op would produce)
    x = x * 1.0901
    x = x.to(tl.bfloat16).to(tl.float32)

    # relu
    x = tl.maximum(x, 0.0)

    # exact gelu (erf-based), computed in fp32 then rounded to bf16
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # softmax over the row (fp32 accumulation, bf16 output)
    g = tl.where(mask, g, float('-inf'))
    m = tl.max(g, axis=0)
    e = tl.exp(g - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.bfloat16)

    tl.store(Y + row * N + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x @ self.W0
            x = x * 1.0901
            x = torch.relu(x)
            x = F.gelu(x)
            return torch.softmax(x, dim=-1)

        # GEMM via cuBLAS (tensor cores on A100)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()

        Mrows, N = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 1024 else 4

        _fused_scale_relu_gelu_softmax[(Mrows,)](
            h, y, N,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y


def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(M, D, generator=g).to(DT)]

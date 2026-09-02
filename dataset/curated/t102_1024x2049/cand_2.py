import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 102
M, D, DT = 1024, 2049, torch.float16


@triton.jit
def _fused_bias_dsoftmax_scale(
    X_ptr, B_ptr, O_ptr,
    N, stride_x, stride_o,
    S1: tl.constexpr, S2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=float('-inf')).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # bias add (fp32 compute, round to fp16 like PyTorch's opmath, then back)
    x = (x + b).to(tl.float16).to(tl.float32)
    x = tl.where(mask, x, float('-inf'))

    # softmax #1 (fp32 accumulate, fp16 intermediate like reference)
    m1 = tl.max(x, 0)
    e1 = tl.exp(x - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, 0)
    p = (e1 / s1).to(tl.float16).to(tl.float32)

    # softmax #2
    p_m = tl.where(mask, p, float('-inf'))
    m2 = tl.max(p_m, 0)
    e2 = tl.exp(p - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, 0)
    q = (e2 / s2).to(tl.float16).to(tl.float32)

    # sequential scalar multiplies with fp16 rounding after each (matches PyTorch)
    q = (q * S1).to(tl.float16).to(tl.float32)
    q = (q * S2).to(tl.float16)

    tl.store(O_ptr + row * stride_o + offs, q, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 4096, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = x @ self.W0
            y = y + self.b1
            y = torch.softmax(y, dim=-1)
            y = torch.softmax(y, dim=-1)
            y = y * 1.3677
            y = y * 1.255
            return y

        # GEMM via cuBLAS (tensor cores)
        logits = torch.matmul(x, self.W0)

        Mrows, N = logits.shape
        out = torch.empty_like(logits)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_bias_dsoftmax_scale[(Mrows,)](
            logits, self.b1, out,
            N, logits.stride(0), out.stride(0),
            1.3677, 1.255,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out

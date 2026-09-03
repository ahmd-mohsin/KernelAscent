import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 400010
S, D, N, E, DT = 1024, 1024, 2048, 4, torch.float16


@triton.jit
def _combine_gelu_kernel(
    outs_ptr, gate_ptr, y_ptr,
    S, N,
    E: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs_n < N

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for e in tl.static_range(E):
        g = tl.load(gate_ptr + pid_s * E + e).to(tl.float32)
        o = tl.load(outs_ptr + e * S * N + pid_s * N + offs_n,
                    mask=mask, other=0.0).to(tl.float32)
        # multiply, round to fp16 (matches reference elementwise fp16 product),
        # then accumulate in fp32 (matches torch half-sum float accumulation)
        p = (g * o).to(tl.float16)
        acc += p.to(tl.float32)

    # reference: sum result stored as fp16, then gelu computed in fp32 opmath
    s16 = acc.to(tl.float16).to(tl.float32)
    out = 0.5 * s16 * (1.0 + tl.math.erf(s16 * 0.7071067811865476))
    tl.store(y_ptr + pid_s * N + offs_n, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wr = nn.Parameter((torch.randn(D, E, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.We = nn.Parameter((torch.randn(E, D, N, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        s, d = x.shape
        e, _, n = self.We.shape

        # gating (tiny matmul + softmax; softmax internally in fp32 like reference)
        gate = torch.softmax(x @ self.Wr, dim=-1).contiguous()

        # single batched GEMM instead of E separate GEMMs + stack
        outs = torch.matmul(x, self.We)  # (E, S, N)
        outs = outs.contiguous()

        y = torch.empty((s, n), device=x.device, dtype=torch.float16)

        BLOCK_N = 1024
        grid = (s, triton.cdiv(n, BLOCK_N))
        _combine_gelu_kernel[grid](
            outs, gate, y,
            s, n,
            E=e,
            BLOCK_N=BLOCK_N,
            num_warps=4,
        )
        return y

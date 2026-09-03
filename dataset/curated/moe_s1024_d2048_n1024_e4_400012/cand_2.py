import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 400012
S, D, N, E, DT = 1024, 2048, 1024, 4, torch.float16


@triton.jit
def _fused_gate_sum_gelu(
    outs_ptr,  # (S, E*N) fp16, contiguous: outs[s, e*N + n]
    gate_ptr,  # (S, E) fp16, contiguous
    y_ptr,     # (S, N) fp16
    total,     # S * N
    N,
    E: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    s = offs // N
    n = offs % N

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    base = s * (E * N) + n
    for e in tl.static_range(E):
        g = tl.load(gate_ptr + s * E + e, mask=mask, other=0.0).to(tl.float32)
        o = tl.load(outs_ptr + base + e * N, mask=mask, other=0.0).to(tl.float32)
        acc += g * o

    # exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    r = 0.5 * acc * (1.0 + tl.math.erf(acc * 0.7071067811865476))
    tl.store(y_ptr + offs, r.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wr = nn.Parameter((torch.randn(D, E, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.We = nn.Parameter((torch.randn(E, D, N, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        E_, D_, N_ = self.We.shape
        S_ = x.shape[0]

        # Cache the fused expert weight matrix (D, E*N) so that all expert
        # matmuls collapse into a single large GEMM on tensor cores.
        W = getattr(self, "_W_fused", None)
        if W is None or W.device != x.device:
            W = self.We.permute(1, 0, 2).reshape(D_, E_ * N_).contiguous()
            self._W_fused = W

        # Gating: small GEMM + softmax (softmax internally uses fp32 accum).
        gate = torch.softmax(x @ self.Wr, dim=-1).contiguous()

        # One big GEMM producing all expert outputs: (S, E*N)
        outs = x @ W

        y = torch.empty((S_, N_), device=x.device, dtype=torch.float16)
        total = S_ * N_
        BLOCK = 1024
        grid = (triton.cdiv(total, BLOCK),)
        _fused_gate_sum_gelu[grid](
            outs, gate, y, total, N_,
            E=E_, BLOCK=BLOCK,
            num_warps=4,
        )
        return y

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 400008
S, D, N, E, DT = 1024, 1024, 1024, 4, torch.float16


@triton.jit
def _moe_gelu_kernel(
    outs_ptr,   # (S, E*N) fp16, row-major, layout [s, e*N + n]
    gate_ptr,   # (S, E) fp16
    y_ptr,      # (S, N) fp16
    N,          # int
    E: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    blk = tl.program_id(1)
    offs = blk * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    base = outs_ptr + row * (E * N)
    gbase = gate_ptr + row * E

    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for e in tl.static_range(E):
        g = tl.load(gbase + e).to(tl.float32)
        o = tl.load(base + e * N + offs, mask=mask, other=0.0).to(tl.float32)
        # replicate fp16 elementwise-mul rounding, then fp32 accumulation (as torch.sum does)
        p = (g * o).to(tl.float16).to(tl.float32)
        acc += p

    # exact (erf-based) GELU in fp32, as PyTorch does for half inputs
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    y = acc * 0.5 * (1.0 + tl.math.erf(acc * INV_SQRT2))

    tl.store(y_ptr + row * N + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wr = nn.Parameter((torch.randn(D, E, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.We = nn.Parameter((torch.randn(E, D, N, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        # Cache experts fused into a single (D, E*N) matrix so all expert matmuls
        # become one big GEMM (each expert occupies a contiguous column block,
        # so per-expert results are bitwise identical to separate GEMMs' math).
        W_fused = getattr(self, "_W_fused", None)
        if W_fused is None or W_fused.device != x.device:
            e, d, n = self.We.shape
            W_fused = self.We.to(x.device).permute(1, 0, 2).reshape(d, e * n).contiguous()
            self._W_fused = W_fused

        x = x.contiguous()
        s = x.shape[0]
        e = self.Wr.shape[1]
        n = W_fused.shape[1] // e

        # gating (tiny GEMM + softmax over E=4)
        gate = torch.softmax(x @ self.Wr, dim=-1).contiguous()  # (S, E) fp16

        # all expert outputs in one GEMM: (S, D) @ (D, E*N) -> (S, E*N)
        outs = x @ W_fused  # (S, E*N), row s holds [e0 block | e1 block | ...]

        y = torch.empty((s, n), device=x.device, dtype=torch.float16)

        BLOCK = 1024
        grid = (s, triton.cdiv(n, BLOCK))
        _moe_gelu_kernel[grid](
            outs, gate, y, n,
            E=e, BLOCK=BLOCK,
            num_warps=8,
        )
        return y

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 400011
S, D, N, E, DT = 1024, 1024, 2048, 8, torch.float16


@triton.jit
def _moe_combine_gelu_kernel(
    outs_ptr,   # [S, E*N] fp16, expert-major columns (e*N + n)
    gate_ptr,   # [S, E]   fp16
    y_ptr,      # [S, N]   fp16
    N,
    E: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N

    base = outs_ptr + pid_s * (E * N)
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    # Match reference: fp16 elementwise multiply, fp32 accumulation over experts
    for e in tl.static_range(E):
        g = tl.load(gate_ptr + pid_s * E + e)                    # fp16 scalar
        o = tl.load(base + e * N + offs, mask=mask, other=0.0)   # fp16
        acc += (g * o).to(tl.float32)

    # Exact (erf-based) GELU computed in fp32, matching PyTorch half GELU semantics
    r = acc * 0.5 * (1.0 + tl.math.erf(acc * 0.7071067811865476))
    tl.store(y_ptr + pid_s * N + offs, r.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wr = nn.Parameter((torch.randn(D, E, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.We = nn.Parameter((torch.randn(E, D, N, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        # One-time flatten of expert weights into a single [D, E*N] matrix so that
        # all E expert matmuls fuse into a single large GEMM (much better tensor-core
        # utilization on A100 than E separate GEMMs). Each output column is the same
        # dot product over D as in the reference, so results are numerically identical.
        if not hasattr(self, "_Wflat"):
            self._Wflat = self.We.permute(1, 0, 2).reshape(self.We.shape[1], -1).contiguous()

        s = x.shape[0]
        n = self.We.shape[2]
        e = self.We.shape[0]

        # Gating: tiny [S, E] matmul + softmax (softmax on half accumulates in fp32,
        # same as the reference implementation).
        gate = torch.softmax(x @ self.Wr, dim=-1).contiguous()

        # Single fused GEMM producing all expert outputs: [S, E*N]
        outs = x @ self._Wflat

        # Fused (gate-weighted expert sum + GELU) Triton kernel — avoids materializing
        # the stacked [E, S, N] tensor and the intermediate y tensor.
        y = torch.empty((s, n), device=x.device, dtype=torch.float16)
        BLOCK_N = 512
        grid = (s, triton.cdiv(n, BLOCK_N))
        _moe_combine_gelu_kernel[grid](
            outs, gate, y, n,
            E=e, BLOCK_N=BLOCK_N,
            num_warps=4,
        )
        return y

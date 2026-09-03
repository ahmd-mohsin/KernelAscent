import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 400006
S, D, N, E, DT = 512, 2048, 2048, 4, torch.float16


@triton.jit
def _gate_sum_gelu_kernel(
    outs_ptr,       # (S, E, N) fp16, contiguous
    gate_ptr,       # (S, E) fp16, contiguous
    y_ptr,          # (S, N) fp16, contiguous
    N,
    E: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs_n < N

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    base = outs_ptr + pid_s * E * N
    for e in tl.static_range(E):
        g = tl.load(gate_ptr + pid_s * E + e).to(tl.float32)
        v = tl.load(base + e * N + offs_n, mask=mask, other=0.0).to(tl.float32)
        acc += g * v

    # exact (erf-based) GELU, computed in fp32 like PyTorch's opmath for half
    out = acc * 0.5 * (1.0 + tl.math.erf(acc * 0.7071067811865476))

    tl.store(y_ptr + pid_s * N + offs_n, out.to(y_ptr.dtype.element_ty), mask=mask)


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

        # Lazily cache a (D, E*N) view of the expert weights so all expert
        # matmuls collapse into a single large tensor-core GEMM.
        We_flat = getattr(self, "_We_flat", None)
        if We_flat is None or We_flat.device != x.device:
            We_flat = self.We.permute(1, 0, 2).reshape(d, e * n).contiguous()
            self._We_flat = We_flat

        # Router gate: tiny GEMM + softmax
        gate = torch.softmax(x @ self.Wr, dim=-1).contiguous()  # (S, E)

        # All experts in one GEMM: (S, D) @ (D, E*N) -> (S, E*N) viewed as (S, E, N)
        outs = x @ We_flat  # (S, E*N), contiguous

        y = torch.empty((s, n), device=x.device, dtype=x.dtype)

        BLOCK_N = 1024
        grid = (s, triton.cdiv(n, BLOCK_N))
        _gate_sum_gelu_kernel[grid](
            outs, gate, y,
            n,
            E=e,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return y

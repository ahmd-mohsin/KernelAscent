import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 400005
S, D, N, E, DT = 512, 2048, 1024, 8, torch.float16


@triton.jit
def _wsum_gelu_kernel(
    gate_ptr, outs_ptr, y_ptr,
    S, N,
    E: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    base = pid_s * N + offs
    for e in tl.static_range(E):
        g = tl.load(gate_ptr + pid_s * E + e)  # fp16 scalar
        o = tl.load(outs_ptr + e * S * N + base, mask=mask, other=0.0)  # fp16
        p = (g * o).to(tl.float16)  # match fp16 elementwise product of reference
        acc += p.to(tl.float32)     # match fp32 accumulation of torch.sum on half

    # exact GELU (erf form), computed in fp32 as PyTorch does internally
    y = 0.5 * acc * (1.0 + tl.math.erf(acc * 0.7071067811865476))
    tl.store(y_ptr + base, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wr = nn.Parameter((torch.randn(D, E, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.We = nn.Parameter((torch.randn(E, D, N, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        s, d = x.shape
        e, _, n = self.We.shape

        # gate: (S, E)
        gate = torch.softmax(x @ self.Wr, dim=-1).contiguous()

        # Batched GEMM: (E, S, N) — replaces the Python loop + stack
        outs = torch.matmul(x, self.We)  # broadcasts x over expert dim

        y = torch.empty((s, n), device=x.device, dtype=x.dtype)

        BLOCK_N = 1024
        grid = (s, triton.cdiv(n, BLOCK_N))
        _wsum_gelu_kernel[grid](
            gate, outs, y,
            s, n,
            E=e,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return y

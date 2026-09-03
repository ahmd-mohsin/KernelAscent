import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 400010
S, D, N, E, DT = 1024, 1024, 2048, 4, torch.float16


@triton.jit
def _moe_gelu_kernel(
    gate_ptr,       # (S, E) fp16
    outs_ptr,       # (S, E*N) fp16, col index = e*N + n
    y_ptr,          # (S, N) fp16
    total,          # S * N
    N_dim,
    EN,             # E * N
    E_dim: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    s = offs // N_dim
    n = offs % N_dim

    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for e in tl.static_range(E_dim):
        g = tl.load(gate_ptr + s * E_dim + e, mask=mask, other=0.0)  # fp16
        o = tl.load(outs_ptr + s * EN + e * N_dim + n, mask=mask, other=0.0)  # fp16
        # match reference: fp16 multiply (rounded), fp32 accumulate (torch.sum acc type)
        prod = g * o
        acc += prod.to(tl.float32)

    # round the sum back to fp16 (as torch.sum would produce a fp16 tensor)
    y16 = acc.to(tl.float16)
    # exact GELU (erf-based), computed in fp32 like torch's opmath for half
    xf = y16.to(tl.float32)
    out = 0.5 * xf * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    tl.store(y_ptr + offs, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wr = nn.Parameter((torch.randn(D, E, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.We = nn.Parameter((torch.randn(E, D, N, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        # Lazily cache the fused expert weight matrix (D, E*N)
        W2 = getattr(self, "_W2", None)
        if W2 is None or W2.device != x.device:
            W2 = self.We.permute(1, 0, 2).reshape(D, E * N).contiguous()
            self._W2 = W2

        # gating (small GEMM + softmax)
        gate = torch.softmax(x @ self.Wr, dim=-1).contiguous()  # (S, E) fp16

        # single big GEMM for all experts: (S, D) @ (D, E*N) -> (S, E*N)
        outs = x @ W2

        y = torch.empty((x.shape[0], N), device=x.device, dtype=torch.float16)
        total = x.shape[0] * N
        BLOCK = 1024
        grid = (triton.cdiv(total, BLOCK),)
        _moe_gelu_kernel[grid](
            gate, outs, y,
            total, N, E * N,
            E_dim=E, BLOCK=BLOCK,
            num_warps=4,
        )
        return y

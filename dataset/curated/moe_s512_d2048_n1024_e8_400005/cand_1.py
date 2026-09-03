import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 400005
S, D, N, E, DT = 512, 2048, 1024, 8, torch.float16


@triton.jit
def _moe_gate_gelu_kernel(
    outs_ptr,   # (E, S, N) fp16, contiguous
    gate_ptr,   # (S, E) fp16, contiguous
    y_ptr,      # (S, N) fp16, contiguous
    S_dim, N_dim,
    E_dim: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs_n < N_dim

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for e in tl.static_range(E_dim):
        g = tl.load(gate_ptr + pid_s * E_dim + e).to(tl.float32)
        o = tl.load(outs_ptr + (e * S_dim + pid_s) * N_dim + offs_n,
                    mask=mask, other=0.0).to(tl.float32)
        acc += g * o

    # exact (erf-based) GELU
    out = 0.5 * acc * (1.0 + tl.math.erf(acc * 0.7071067811865476))
    tl.store(y_ptr + pid_s * N_dim + offs_n,
             out.to(y_ptr.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wr = nn.Parameter((torch.randn(D, E, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.We = nn.Parameter((torch.randn(E, D, N, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            gate = torch.softmax(x @ self.Wr, dim=-1)
            outs = torch.stack([x @ self.We[e] for e in range(E)], dim=0)
            y = (gate.transpose(0, 1).unsqueeze(-1) * outs).sum(0)
            return F.gelu(y)

        x = x.contiguous()
        s, d = x.shape
        e_dim, _, n = self.We.shape

        # gating (tiny matmul + softmax over E=8)
        gate = torch.softmax(x @ self.Wr, dim=-1).contiguous()  # (S, E)

        # one batched GEMM instead of E separate GEMMs + stack copy
        outs = torch.matmul(x, self.We)  # (E, S, N), contiguous
        if not outs.is_contiguous():
            outs = outs.contiguous()

        y = torch.empty((s, n), device=x.device, dtype=x.dtype)

        BLOCK_N = 1024 if n >= 1024 else triton.next_power_of_2(n)
        grid = (s, triton.cdiv(n, BLOCK_N))
        _moe_gate_gelu_kernel[grid](
            outs, gate, y,
            s, n,
            E_dim=e_dim,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return y

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 400007
S, D, N, E, DT = 512, 2048, 2048, 8, torch.float16


@triton.jit
def _weighted_sum_gelu_kernel(
    outs_ptr, gate_ptr, y_ptr,
    S_dim, N_dim,
    E_dim: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs = pid_n * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_dim

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    base = pid_s * N_dim + offs
    for e in tl.static_range(E_dim):
        g = tl.load(gate_ptr + pid_s * E_dim + e).to(tl.float32)
        v = tl.load(outs_ptr + e * S_dim * N_dim + base, mask=mask, other=0.0).to(tl.float32)
        acc += g * v

    # exact (erf-based) GELU in fp32, matching F.gelu on half (opmath = float)
    out = acc * 0.5 * (1.0 + tl.math.erf(acc * 0.7071067811865476))
    tl.store(y_ptr + base, out.to(tl.float16), mask=mask)


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
        s_dim, d_dim = x.shape
        e_dim = self.We.shape[0]
        n_dim = self.We.shape[2]

        # gate: (S, E) — small GEMM + softmax (softmax internally uses fp32 accumulation)
        gate = torch.softmax(x @ self.Wr, dim=-1).contiguous()

        # one batched GEMM instead of E separate GEMMs: (E, S, N)
        outs = torch.matmul(x, self.We)  # broadcasts x over batch dim -> (E, S, N)
        outs = outs.contiguous()

        y = torch.empty((s_dim, n_dim), device=x.device, dtype=torch.float16)

        BLOCK = 1024
        grid = (s_dim, triton.cdiv(n_dim, BLOCK))
        _weighted_sum_gelu_kernel[grid](
            outs, gate, y,
            s_dim, n_dim,
            E_dim=e_dim,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y

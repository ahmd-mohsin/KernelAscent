import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl

SEED = 400003
S, D, N, E, DT = 512, 1024, 2048, 8, torch.float16


@triton.jit
def _wsum_gelu_kernel(
    outs_ptr,  # (E, S, N) fp16
    gate_ptr,  # (S, E) fp16
    y_ptr,     # (S, N) fp16
    S_dim, N_dim,
    E_dim: tl.constexpr,
    BLOCK: tl.constexpr,
):
    s = tl.program_id(0)
    nb = tl.program_id(1)
    offs = nb * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_dim

    acc = tl.zeros([BLOCK], dtype=tl.float32)
    base = s * N_dim + offs
    for e in tl.static_range(E_dim):
        g = tl.load(gate_ptr + s * E_dim + e).to(tl.float32)
        o = tl.load(outs_ptr + e * S_dim * N_dim + base, mask=mask, other=0.0).to(tl.float32)
        acc += g * o

    # match reference: sum result cast to fp16, then gelu computed in fp32
    y16 = acc.to(tl.float16)
    yf = y16.to(tl.float32)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    out = 0.5 * yf * (1.0 + tl.math.erf(yf * INV_SQRT2))
    tl.store(y_ptr + base, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wr = nn.Parameter((torch.randn(D, E, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.We = nn.Parameter((torch.randn(E, D, N, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        # gate: (S, E)
        gate = torch.softmax(x @ self.Wr, dim=-1)
        # batched matmul: (E, S, N)  -- one cuBLAS strided-batched GEMM instead of E separate GEMMs
        outs = torch.matmul(x.unsqueeze(0), self.We)

        s_dim, n_dim = x.shape[0], self.We.shape[2]
        e_dim = self.We.shape[0]

        if x.is_cuda:
            outs = outs.contiguous()
            gate = gate.contiguous()
            y = torch.empty((s_dim, n_dim), device=x.device, dtype=torch.float16)
            BLOCK = 1024
            grid = (s_dim, triton.cdiv(n_dim, BLOCK))
            _wsum_gelu_kernel[grid](
                outs, gate, y,
                s_dim, n_dim,
                E_dim=e_dim,
                BLOCK=BLOCK,
                num_warps=8,
            )
            return y
        else:
            y = (gate.transpose(0, 1).unsqueeze(-1) * outs).sum(0)
            return F.gelu(y)

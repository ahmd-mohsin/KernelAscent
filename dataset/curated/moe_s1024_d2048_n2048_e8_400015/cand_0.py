import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 400015
S, D, N, E, DT = 1024, 2048, 2048, 8, torch.float16


@triton.jit
def _moe_combine_gelu_kernel(
    outs_ptr,  # [E, S, N] fp16
    gate_ptr,  # [S, E] fp16
    y_ptr,     # [S, N] fp16
    S_dim, N_dim,
    E_dim: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs = pid_n * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_dim

    acc = tl.zeros([BLOCK], dtype=tl.float32)
    base = pid_s * N_dim + offs
    for e in tl.static_range(E_dim):
        g = tl.load(gate_ptr + pid_s * E_dim + e).to(tl.float32)
        o = tl.load(outs_ptr + e * S_dim * N_dim + base, mask=mask, other=0.0).to(tl.float32)
        acc += g * o

    # exact (erf-based) GELU, matching F.gelu default
    y = 0.5 * acc * (1.0 + tl.math.erf(acc * 0.7071067811865476))
    tl.store(y_ptr + base, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wr = nn.Parameter((torch.randn(D, E, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.We = nn.Parameter((torch.randn(E, D, N, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        # gate: [S, E]
        gate = torch.softmax(x @ self.Wr, dim=-1).contiguous()

        # all expert outputs in ONE batched GEMM: [E, S, N]
        outs = torch.matmul(x.unsqueeze(0), self.We)
        outs = outs.contiguous()

        s, n = x.shape[0], outs.shape[-1]
        e = self.We.shape[0]
        y = torch.empty((s, n), device=x.device, dtype=x.dtype)

        BLOCK = 512
        grid = (s, triton.cdiv(n, BLOCK))
        _moe_combine_gelu_kernel[grid](
            outs, gate, y,
            s, n,
            E_dim=e,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return y

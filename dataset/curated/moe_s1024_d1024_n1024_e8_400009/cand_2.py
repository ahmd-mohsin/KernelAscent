import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 400009
S, D, N, E, DT = 1024, 1024, 1024, 8, torch.float16


@triton.jit
def _moe_gate_gelu_kernel(
    logits_ptr,      # (S, E) fp16 gating logits
    outs_ptr,        # (S, E, N) fp16 expert outputs
    y_ptr,           # (S, N) fp16 output
    N,
    E: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_n = tl.program_id(1)

    # ---- softmax over the E gating logits (fp32) ----
    e_offs = tl.arange(0, E)
    lg = tl.load(logits_ptr + pid_s * E + e_offs).to(tl.float32)
    m = tl.max(lg, axis=0)
    p = tl.exp(lg - m)
    gate = p / tl.sum(p, axis=0)  # (E,)

    # ---- weighted sum over experts for a tile of N ----
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N

    ptrs = outs_ptr + pid_s * E * N + e_offs[:, None] * N + n_offs[None, :]
    o = tl.load(ptrs, mask=n_mask[None, :], other=0.0).to(tl.float32)  # (E, BLOCK_N)

    y = tl.sum(gate[:, None] * o, axis=0)  # (BLOCK_N,) fp32

    # ---- exact (erf-based) GELU ----
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    y = 0.5 * y * (1.0 + tl.math.erf(y * INV_SQRT2))

    tl.store(y_ptr + pid_s * N + n_offs, y.to(tl.float16), mask=n_mask)


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

        # Cache flattened expert weights: (D, E*N) so all experts run in ONE GEMM
        We_flat = getattr(self, "_We_flat", None)
        if We_flat is None or We_flat.device != x.device:
            We_flat = self.We.permute(1, 0, 2).reshape(D, E * N).contiguous()
            self._We_flat = We_flat

        s_dim = x.shape[0]

        # Gating logits (softmax fused into the Triton kernel)
        logits = x @ self.Wr                       # (S, E) fp16

        # All expert outputs in one tensor-core GEMM: (S, E*N)
        outs = x @ We_flat                         # (S, E*N) fp16, laid out as (S, E, N)

        y = torch.empty((s_dim, N), device=x.device, dtype=torch.float16)

        BLOCK_N = 256
        grid = (s_dim, triton.cdiv(N, BLOCK_N))
        _moe_gate_gelu_kernel[grid](
            logits, outs, y,
            N,
            E=E,
            BLOCK_N=BLOCK_N,
            num_warps=4,
        )
        return y

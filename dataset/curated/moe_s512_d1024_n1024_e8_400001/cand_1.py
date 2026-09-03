import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 400001
S, D, N, E, DT = 512, 1024, 1024, 8, torch.float16


@triton.jit
def _fused_gate_sum_gelu(
    logits_ptr, outs_ptr, y_ptr,
    N, EN,
    BLOCK: tl.constexpr, E: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_n = tl.program_id(1)

    # ---- softmax over E logits (fp32 math, round gate to fp16 like reference) ----
    offs_e = tl.arange(0, E)
    lg = tl.load(logits_ptr + pid_s * E + offs_e).to(tl.float32)
    m = tl.max(lg, 0)
    ex = tl.exp(lg - m)
    w = ex / tl.sum(ex, 0)
    w = w.to(tl.float16).to(tl.float32)

    # ---- weighted sum over experts + exact GELU ----
    offs_n = pid_n * BLOCK + tl.arange(0, BLOCK)
    mask = offs_n < N
    base = outs_ptr + pid_s * EN

    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for e in tl.static_range(E):
        v = tl.load(base + e * N + offs_n, mask=mask, other=0.0).to(tl.float32)
        we = tl.sum(tl.where(offs_e == e, w, 0.0), 0)
        # product rounded to fp16 (matches gate*outs in fp16), fp32 accumulation
        p = (we * v).to(tl.float16).to(tl.float32)
        acc += p

    y = acc.to(tl.float16).to(tl.float32)
    g = 0.5 * y * (1.0 + tl.math.erf(y * 0.7071067811865476))
    tl.store(y_ptr + pid_s * N + offs_n, g.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wr = nn.Parameter((torch.randn(D, E, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.We = nn.Parameter((torch.randn(E, D, N, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        Sx, Dx = x.shape

        # Lazily cache the flattened expert weights: (D, E*N) so all expert
        # matmuls become a single large GEMM on tensor cores.
        Wflat = getattr(self, "_Wflat", None)
        if Wflat is None or Wflat.device != x.device:
            Wflat = self.We.permute(1, 0, 2).reshape(Dx, E * N).contiguous()
            self._Wflat = Wflat

        # gating logits (small GEMM) and expert outputs (one big GEMM)
        logits = x @ self.Wr                      # (S, E)
        outs = (x @ Wflat).contiguous()           # (S, E*N), row s holds [e0 | e1 | ... ]

        y = torch.empty((Sx, N), device=x.device, dtype=torch.float16)

        BLOCK = 1024
        grid = (Sx, triton.cdiv(N, BLOCK))
        _fused_gate_sum_gelu[grid](
            logits, outs, y,
            N, E * N,
            BLOCK=BLOCK, E=E,
            num_warps=8,
        )
        return y

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 400001
S, D, N, E, DT = 512, 1024, 1024, 8, torch.float16


@triton.jit
def _fused_moe_gelu_kernel(
    logits_ptr,   # (S, E) fp16 gating logits
    outs_ptr,     # (S, E, N) fp16 expert outputs
    y_ptr,        # (S, N) fp16 output
    N,            # runtime N
    E: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)

    # ---- softmax over the E gating logits (fp32) ----
    offs_e = tl.arange(0, E)
    lg = tl.load(logits_ptr + row * E + offs_e).to(tl.float32)
    m = tl.max(lg, axis=0)
    p = tl.exp(lg - m)
    p = p / tl.sum(p, axis=0)

    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    # ---- weighted sum over experts ----
    base = row * E * N
    ptrs = outs_ptr + base + offs_e[:, None] * N + offs_n[None, :]
    o = tl.load(ptrs, mask=mask_n[None, :], other=0.0).to(tl.float32)
    acc = tl.sum(p[:, None] * o, axis=0)

    # ---- exact GELU (erf) ----
    g = 0.5 * acc * (1.0 + tl.math.erf(acc * 0.7071067811865476))

    tl.store(y_ptr + row * N + offs_n, g.to(tl.float16), mask=mask_n)


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
        e, _, n = self.We.shape

        # Lazily cache flattened expert weights: (D, E*N) so a single GEMM
        # produces all expert outputs with layout (S, E, N).
        We_flat = getattr(self, "_We_flat", None)
        if We_flat is None or We_flat.device != x.device:
            We_flat = self.We.permute(1, 0, 2).reshape(d, e * n).contiguous()
            self._We_flat = We_flat

        # Gating logits: (S, E)
        logits = (x @ self.Wr).contiguous()

        # All expert outputs in one large GEMM: (S, E*N) viewed as (S, E, N)
        outs = x @ We_flat  # contiguous (S, E*N)

        y = torch.empty((s, n), device=x.device, dtype=torch.float16)

        BLOCK_N = triton.next_power_of_2(n)
        grid = (s,)
        _fused_moe_gelu_kernel[grid](
            logits, outs, y, n,
            E=e, BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return y

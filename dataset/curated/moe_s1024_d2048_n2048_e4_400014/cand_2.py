import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 400014
S, D, N, E, DT = 1024, 2048, 2048, 4, torch.float16


@triton.jit
def _moe_combine_gelu_kernel(
    outs_ptr,   # (S, E*N) fp16
    gate_ptr,   # (S, E) fp16
    y_ptr,      # (S, N) fp16
    N,          # int
    E: tl.constexpr,
    BLOCK: tl.constexpr,
):
    s = tl.program_id(0)
    nb = tl.program_id(1)
    offs = nb * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    acc = tl.zeros([BLOCK], dtype=tl.float32)
    base = s * E * N
    for e in tl.static_range(E):
        g = tl.load(gate_ptr + s * E + e).to(tl.float32)
        o = tl.load(outs_ptr + base + e * N + offs, mask=mask, other=0.0).to(tl.float32)
        acc += g * o

    # exact GELU (erf form) to match F.gelu default
    out = acc * 0.5 * (1.0 + tl.math.erf(acc * 0.70710678118654752440))
    tl.store(y_ptr + s * N + offs, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wr = nn.Parameter((torch.randn(D, E, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.We = nn.Parameter((torch.randn(E, D, N, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        s, d = x.shape
        e, _, n = self.We.shape

        # Lazily cache the experts flattened into a single (D, E*N) matrix
        # so all expert matmuls become one big tensor-core GEMM.
        We_flat = getattr(self, "_We_flat", None)
        if We_flat is None or We_flat.device != x.device:
            We_flat = self.We.permute(1, 0, 2).contiguous().view(d, e * n)
            self._We_flat = We_flat

        # Router gate: small GEMM + softmax over E (=4)
        gate = torch.softmax(x @ self.Wr, dim=-1).contiguous()  # (S, E) fp16

        # All expert outputs in one GEMM: (S, D) @ (D, E*N) -> (S, E*N)
        outs = x @ We_flat  # (S, E*N) fp16

        y = torch.empty((s, n), device=x.device, dtype=x.dtype)

        BLOCK = 1024
        grid = (s, triton.cdiv(n, BLOCK))
        _moe_combine_gelu_kernel[grid](
            outs, gate, y, n,
            E=e, BLOCK=BLOCK,
            num_warps=8,
        )
        return y

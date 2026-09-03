import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 400012
S, D, N, E, DT = 1024, 2048, 1024, 4, torch.float16


@triton.jit
def _gate_sum_gelu_kernel(
    outs_ptr,   # (S, E, N) fp16, contiguous
    gate_ptr,   # (S, E) fp16/fp32
    y_ptr,      # (S, N) fp16
    N,
    E: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cb = tl.program_id(1)
    n = cb * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = n < N

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    base = outs_ptr + row * (E * N)
    for e in tl.static_range(E):
        g = tl.load(gate_ptr + row * E + e).to(tl.float32)
        o = tl.load(base + e * N + n, mask=mask, other=0.0).to(tl.float32)
        acc += g * o

    # exact (erf-based) GELU
    y = acc * 0.5 * (1.0 + tl.math.erf(acc * 0.7071067811865476))
    tl.store(y_ptr + row * N + n, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wr = nn.Parameter((torch.randn(D, E, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.We = nn.Parameter((torch.randn(E, D, N, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self._We_flat = None  # cached (D, E*N) layout for a single big GEMM

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback: reference computation
            gate = torch.softmax(x @ self.Wr, dim=-1)
            outs = torch.stack([x @ self.We[e] for e in range(E)], dim=0)
            y = (gate.transpose(0, 1).unsqueeze(-1) * outs).sum(0)
            return F.gelu(y)

        if self._We_flat is None or self._We_flat.device != self.We.device:
            # (E, D, N) -> (D, E, N) -> (D, E*N), done once and cached
            self._We_flat = self.We.permute(1, 0, 2).reshape(self.We.shape[1], -1).contiguous()

        x = x.contiguous()
        s = x.shape[0]

        # gate: (S, E)
        gate = torch.softmax(x @ self.Wr, dim=-1).contiguous()

        # single large GEMM: (S, D) @ (D, E*N) -> (S, E*N), viewed as (S, E, N)
        outs = x @ self._We_flat

        y = torch.empty((s, N), device=x.device, dtype=torch.float16)

        BLOCK_N = 1024
        grid = (s, triton.cdiv(N, BLOCK_N))
        _gate_sum_gelu_kernel[grid](
            outs, gate, y, N,
            E=E, BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return y

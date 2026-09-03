import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 400011
S, D, N, E, DT = 1024, 1024, 2048, 8, torch.float16


@triton.jit
def _gate_sum_gelu_kernel(
    outs_ptr,   # (S, E, N) fp16 contiguous
    gate_ptr,   # (S, E) fp16 contiguous
    y_ptr,      # (S, N) fp16 contiguous
    N,
    E: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs_n < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    base = outs_ptr + pid_s * E * N

    for e in tl.static_range(E):
        g = tl.load(gate_ptr + pid_s * E + e)                    # fp16 scalar
        o = tl.load(base + e * N + offs_n, mask=mask, other=0.0) # fp16 block
        prod = g * o                                             # fp16 product (matches ref)
        acc += prod.to(tl.float32)                               # fp32 accumulate (matches sum)

    y = acc.to(tl.float16).to(tl.float32)  # round to fp16 like ref, gelu in fp32 opmath
    out = 0.5 * y * (1.0 + tl.math.erf(y * 0.7071067811865476))
    tl.store(y_ptr + pid_s * N + offs_n, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wr = nn.Parameter((torch.randn(D, E, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.We = nn.Parameter((torch.randn(E, D, N, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def _get_we_flat(self):
        cached = getattr(self, "_We_flat", None)
        if (cached is None
                or cached.device != self.We.device
                or cached.dtype != self.We.dtype):
            e, d, n = self.We.shape
            # (E, D, N) -> (D, E*N) so one big GEMM computes all experts at once
            self._We_flat = self.We.permute(1, 0, 2).reshape(d, e * n).contiguous()
            cached = self._We_flat
        return cached

    def forward(self, x):
        e, d, n = self.We.shape
        s = x.shape[0]

        # gating (softmax over E; PyTorch accumulates fp16 softmax in fp32)
        gate = torch.softmax(x @ self.Wr, dim=-1).contiguous()  # (S, E) fp16

        # single large GEMM instead of E separate GEMMs
        we_flat = self._get_we_flat()                # (D, E*N)
        outs = (x @ we_flat).view(s, e, n)           # (S, E, N) fp16, contiguous

        y = torch.empty((s, n), device=x.device, dtype=x.dtype)

        BLOCK_N = 1024
        grid = (s, triton.cdiv(n, BLOCK_N))
        _gate_sum_gelu_kernel[grid](
            outs, gate, y, n,
            E=e, BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return y

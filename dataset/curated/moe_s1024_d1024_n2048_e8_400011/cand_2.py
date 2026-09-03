import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 400011
S, D, N, E, DT = 1024, 1024, 2048, 8, torch.float16


@triton.jit
def _moe_combine_gelu_kernel(
    z_ptr,      # (S, E, N) fp16 contiguous
    g_ptr,      # (S, E) fp16 contiguous
    out_ptr,    # (S, N) fp16 contiguous
    N: tl.constexpr,
    E: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs_n < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    z_base = z_ptr + pid_s * E * N
    g_base = g_ptr + pid_s * E
    for e in tl.static_range(E):
        g = tl.load(g_base + e).to(tl.float32)
        zv = tl.load(z_base + e * N + offs_n, mask=mask, other=0.0).to(tl.float32)
        acc += g * zv

    # exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    INV_SQRT2: tl.constexpr = 0.7071067811865475
    y = 0.5 * acc * (1.0 + tl.math.erf(acc * INV_SQRT2))

    tl.store(out_ptr + pid_s * N + offs_n, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wr = nn.Parameter((torch.randn(D, E, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.We = nn.Parameter((torch.randn(E, D, N, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self._We_flat = None  # lazy cache: (D, E*N)

    def forward(self, x):
        if (self._We_flat is None
                or self._We_flat.device != self.We.device
                or self._We_flat.dtype != self.We.dtype):
            # (E, D, N) -> (D, E, N) -> (D, E*N)
            self._We_flat = self.We.permute(1, 0, 2).reshape(D, E * N).contiguous()

        s = x.shape[0]

        # gate = softmax(x @ Wr)  -> (S, E)
        gate = torch.softmax(x @ self.Wr, dim=-1).contiguous()

        # all experts in one big GEMM: (S, D) @ (D, E*N) -> (S, E*N)
        z = (x @ self._We_flat).contiguous()

        out = torch.empty((s, N), device=x.device, dtype=torch.float16)

        BLOCK_N = 1024
        grid = (s, triton.cdiv(N, BLOCK_N))
        _moe_combine_gelu_kernel[grid](
            z, gate, out,
            N=N, E=E, BLOCK_N=BLOCK_N,
            num_warps=4,
        )
        return out

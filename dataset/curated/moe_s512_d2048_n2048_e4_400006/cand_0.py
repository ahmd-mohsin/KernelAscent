import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 400006
S, D, N, E, DT = 512, 2048, 2048, 4, torch.float16


@triton.jit
def _moe_combine_gelu_kernel(
    outs_ptr,      # (S, E, N) contiguous, fp16
    gate_ptr,      # (S, E) contiguous, fp16
    y_ptr,         # (S, N) contiguous, fp16
    N,
    E: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs_n < N

    base = pid_s * E * N
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for e in tl.static_range(E):
        g = tl.load(gate_ptr + pid_s * E + e)  # fp16 scalar
        o = tl.load(outs_ptr + base + e * N + offs_n, mask=mask, other=0.0)
        # multiply in fp16 (matches reference elementwise multiply),
        # accumulate in fp32 (matches torch sum-reduction accumulation)
        prod = g * o
        acc += prod.to(tl.float32)

    # exact (erf-based) GELU computed in fp32, matching F.gelu opmath
    out = acc * 0.5 * (1.0 + tl.math.erf(acc * 0.7071067811865476))
    tl.store(y_ptr + pid_s * N + offs_n, out.to(y_ptr.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wr = nn.Parameter((torch.randn(D, E, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.We = nn.Parameter((torch.randn(E, D, N, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self._We_flat = None  # lazy cache: (D, E*N)

    def forward(self, x):
        if self._We_flat is None or self._We_flat.device != x.device:
            # (E, D, N) -> (D, E, N) -> (D, E*N), one big GEMM instead of E small ones
            self._We_flat = (
                self.We.to(x.device).permute(1, 0, 2).reshape(self.We.shape[1], -1).contiguous()
            )

        Sx = x.shape[0]
        Nn = self.We.shape[2]
        Ee = self.We.shape[0]

        # gating: small GEMM + softmax
        gate = torch.softmax(x @ self.Wr, dim=-1).contiguous()  # (S, E) fp16

        # one large GEMM for all experts: (S, D) @ (D, E*N) -> (S, E*N)
        outs = x @ self._We_flat  # (S, E*N), row layout = (S, E, N)

        y = torch.empty((Sx, Nn), device=x.device, dtype=x.dtype)

        BLOCK_N = 1024
        grid = (Sx, triton.cdiv(Nn, BLOCK_N))
        _moe_combine_gelu_kernel[grid](
            outs, gate, y, Nn, E=Ee, BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return y

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 400010
S, D, N, E, DT = 1024, 1024, 2048, 4, torch.float16


@triton.jit
def _moe_gelu_kernel(
    gate_ptr, out_ptr, y_ptr,
    N,
    E: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs = pid_n * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    acc = tl.zeros([BLOCK], dtype=tl.float32)
    base = pid_s * E * N
    for e in tl.static_range(E):
        g = tl.load(gate_ptr + pid_s * E + e).to(tl.float32)
        o = tl.load(out_ptr + base + e * N + offs, mask=mask, other=0.0).to(tl.float32)
        acc += g * o

    # exact GELU: 0.5*x*(1+erf(x/sqrt(2)))
    y = 0.5 * acc * (1.0 + tl.math.erf(acc * 0.7071067811865476))
    tl.store(y_ptr + pid_s * N + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wr = nn.Parameter((torch.randn(D, E, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.We = nn.Parameter((torch.randn(E, D, N, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self._W2 = None  # cached fused expert weight (D, E*N)

    def forward(self, x):
        We = self.We
        if self._W2 is None or self._W2.device != We.device:
            # (E, D, N) -> (D, E, N) -> (D, E*N), one big GEMM instead of E small ones
            self._W2 = We.permute(1, 0, 2).reshape(We.shape[1], -1).contiguous()

        e, d, n = We.shape
        s = x.shape[0]

        # gating: tiny GEMM + softmax
        gate = torch.softmax(x @ self.Wr, dim=-1).contiguous()  # (S, E)

        # all experts in a single tensor-core GEMM: (S, D) @ (D, E*N) -> (S, E*N)
        outs = x @ self._W2  # (S, E*N), row-major layout = [s][e][n]

        y = torch.empty((s, n), device=x.device, dtype=torch.float16)

        BLOCK = 1024
        grid = (s, triton.cdiv(n, BLOCK))
        _moe_gelu_kernel[grid](
            gate, outs, y,
            n,
            E=e,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y

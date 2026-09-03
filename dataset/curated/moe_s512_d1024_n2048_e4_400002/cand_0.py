import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 400002
S, D, N, E, DT = 512, 1024, 2048, 4, torch.float16


@triton.jit
def _moe_combine_gelu_kernel(
    gate_ptr,   # (S, E) fp16
    out_ptr,    # (S, E, N) fp16 contiguous
    y_ptr,      # (S, N) fp16
    N,
    E: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs_n < N

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    base = pid_s * E * N
    for e in tl.static_range(E):
        g = tl.load(gate_ptr + pid_s * E + e).to(tl.float32)
        o = tl.load(out_ptr + base + e * N + offs_n, mask=mask, other=0.0).to(tl.float32)
        acc += g * o

    # exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    y = 0.5 * acc * (1.0 + tl.math.erf(acc * 0.7071067811865476))

    tl.store(y_ptr + pid_s * N + offs_n, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wr = nn.Parameter((torch.randn(D, E, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.We = nn.Parameter((torch.randn(E, D, N, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self._Wcat = None  # lazily-built (D, E*N) weight for one big GEMM

    def forward(self, x):
        if self._Wcat is None or self._Wcat.device != x.device:
            # (E, D, N) -> (D, E, N) -> (D, E*N)
            self._Wcat = self.We.permute(1, 0, 2).reshape(self.We.shape[1], -1).contiguous()

        s, d = x.shape
        e = self.We.shape[0]
        n = self.We.shape[2]

        # gate: (S, E) - tiny matmul + softmax
        gate = torch.softmax(x @ self.Wr, dim=-1).contiguous()

        # all experts in a single GEMM: (S, D) @ (D, E*N) -> (S, E, N)
        outs = (x @ self._Wcat).view(s, e, n)

        y = torch.empty((s, n), device=x.device, dtype=x.dtype)

        BLOCK_N = 1024
        grid = (s, triton.cdiv(n, BLOCK_N))
        _moe_combine_gelu_kernel[grid](
            gate, outs, y, n,
            E=e, BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return y

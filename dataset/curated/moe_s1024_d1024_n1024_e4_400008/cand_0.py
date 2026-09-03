import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 400008
S, D, N, E, DT = 1024, 1024, 1024, 4, torch.float16


@triton.jit
def _moe_gelu_kernel(
    OUT_ptr, GATE_ptr, Y_ptr,
    N: tl.constexpr, E: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs_n < N

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    row_base = OUT_ptr + pid_s * (E * N)
    gate_base = GATE_ptr + pid_s * E
    for e in tl.static_range(E):
        g = tl.load(gate_base + e).to(tl.float32)
        v = tl.load(row_base + e * N + offs_n, mask=mask, other=0.0).to(tl.float32)
        acc += g * v

    # exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    res = 0.5 * acc * (1.0 + tl.math.erf(acc * 0.7071067811865476))

    tl.store(Y_ptr + pid_s * N + offs_n, res.to(Y_ptr.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wr = nn.Parameter((torch.randn(D, E, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.We = nn.Parameter((torch.randn(E, D, N, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        e_dim, d_dim, n_dim = self.We.shape
        s_dim = x.shape[0]

        # Cache the flattened expert weight matrix: (D, E*N)
        wflat = getattr(self, "_Wflat", None)
        if wflat is None or wflat.device != x.device:
            wflat = self.We.permute(1, 0, 2).reshape(d_dim, e_dim * n_dim).contiguous()
            self._Wflat = wflat

        # gate: (S, E) softmax over experts (tiny matmul + softmax)
        gate = torch.softmax(x @ self.Wr, dim=-1).contiguous()

        # single large GEMM: (S, D) @ (D, E*N) -> (S, E*N)
        outs = (x @ wflat).contiguous()

        y = torch.empty((s_dim, n_dim), device=x.device, dtype=x.dtype)

        BLOCK_N = 1024
        grid = (s_dim, triton.cdiv(n_dim, BLOCK_N))
        _moe_gelu_kernel[grid](
            outs, gate, y,
            N=n_dim, E=e_dim, BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return y

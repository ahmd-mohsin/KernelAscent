import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 400013
S, D, N, E, DT = 1024, 2048, 1024, 8, torch.float16


@triton.jit
def _gate_reduce_gelu_kernel(
    gate_ptr, big_ptr, out_ptr,
    N,
    E: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs_n < N

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    base = big_ptr + pid_s * E * N
    for e in tl.static_range(E):
        g = tl.load(gate_ptr + pid_s * E + e).to(tl.float32)
        v = tl.load(base + e * N + offs_n, mask=mask, other=0.0).to(tl.float32)
        acc += g * v

    # exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    y = 0.5 * acc * (1.0 + tl.math.erf(acc * 0.7071067811865476))
    tl.store(out_ptr + pid_s * N + offs_n, y.to(out_ptr.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wr = nn.Parameter((torch.randn(D, E, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.We = nn.Parameter((torch.randn(E, D, N, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        # Cache the fused expert weight matrix (D, E*N) so all expert GEMMs
        # run as one large tensor-core matmul.
        We = self.We
        Ee, Dd, Nn = We.shape
        W2 = getattr(self, "_W2", None)
        if W2 is None or self._W2_version != We._version or W2.device != We.device:
            W2 = We.permute(1, 0, 2).reshape(Dd, Ee * Nn).contiguous()
            self._W2 = W2
            self._W2_version = We._version

        # Gating: (S, E)
        gate = torch.softmax(x @ self.Wr, dim=-1).contiguous()

        # All expert outputs in a single GEMM: (S, E*N) viewed as (S, E, N)
        big = x @ W2  # contiguous (S, E*N)

        Ss = x.shape[0]
        out = torch.empty((Ss, Nn), device=x.device, dtype=x.dtype)

        BLOCK_N = 1024
        grid = (Ss, triton.cdiv(Nn, BLOCK_N))
        _gate_reduce_gelu_kernel[grid](
            gate, big, out,
            Nn,
            E=Ee,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return out

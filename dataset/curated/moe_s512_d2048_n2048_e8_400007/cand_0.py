import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 400007
S, D, N, E, DT = 512, 2048, 2048, 8, torch.float16


@triton.jit
def _moe_combine_gelu_kernel(
    gate_ptr,      # (S, E) fp16
    out_ptr,       # (S, E*N) fp16, laid out row-major [s, e, n]
    y_ptr,         # (S, N) fp16
    N,
    E: tl.constexpr,
    BLOCK: tl.constexpr,
):
    s = tl.program_id(0)
    nb = tl.program_id(1)
    offs = nb * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    base = out_ptr + s * E * N
    for e in tl.static_range(E):
        g = tl.load(gate_ptr + s * E + e).to(tl.float32)
        o = tl.load(base + e * N + offs, mask=mask, other=0.0).to(tl.float32)
        acc += g * o

    # exact (erf-based) GELU in fp32, matching F.gelu opmath for half inputs
    y = 0.5 * acc * (1.0 + tl.math.erf(acc * 0.7071067811865476))
    tl.store(y_ptr + s * N + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wr = nn.Parameter((torch.randn(D, E, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.We = nn.Parameter((torch.randn(E, D, N, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        # Lazily build a fused weight matrix (D, E*N) so all expert GEMMs
        # become one large tensor-core GEMM instead of E separate GEMMs.
        W2 = getattr(self, "_W2_cache", None)
        if W2 is None or W2.device != self.We.device:
            W2 = self.We.permute(1, 0, 2).reshape(self.We.shape[1], -1).contiguous()
            self.__dict__["_W2_cache"] = W2

        s = x.shape[0]
        e = self.Wr.shape[1]
        n = self.We.shape[2]

        # Router: small GEMM + softmax
        gate = torch.softmax(x @ self.Wr, dim=-1).contiguous()  # (S, E) fp16

        # All experts in one GEMM: (S, D) @ (D, E*N) -> (S, E*N)
        out = x @ W2  # contiguous, row layout [s, e, n]

        y = torch.empty((s, n), device=x.device, dtype=x.dtype)

        BLOCK = 1024
        grid = (s, triton.cdiv(n, BLOCK))
        _moe_combine_gelu_kernel[grid](
            gate, out, y, n,
            E=e, BLOCK=BLOCK,
            num_warps=8,
        )
        return y

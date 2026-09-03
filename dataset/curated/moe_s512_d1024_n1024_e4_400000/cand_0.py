import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 400000
S, D, N, E, DT = 512, 1024, 1024, 4, torch.float16


@triton.jit
def _moe_combine_gelu_kernel(
    gate_ptr,          # (S, E) fp16
    outs_ptr,          # (S, E*N) fp16
    y_ptr,             # (S, N) fp16
    N: tl.constexpr,
    E: tl.constexpr,
    BLOCK: tl.constexpr,
):
    s = tl.program_id(0)
    nb = tl.program_id(1)
    offs = nb * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    acc = tl.zeros([BLOCK], dtype=tl.float32)
    base = outs_ptr + s * (E * N)
    for e in tl.static_range(E):
        g = tl.load(gate_ptr + s * E + e)
        v = tl.load(base + e * N + offs, mask=mask, other=0.0)
        # multiply in fp16 (matches reference elementwise mul), accumulate fp32 (matches torch.sum)
        p = (g * v).to(tl.float32)
        acc += p

    # exact (erf) GELU in fp32, matching F.gelu opmath behavior on fp16
    out = 0.5 * acc * (1.0 + tl.math.erf(acc * 0.7071067811865476))
    tl.store(y_ptr + s * N + offs, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wr = nn.Parameter((torch.randn(D, E, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.We = nn.Parameter((torch.randn(E, D, N, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        # Lazily cache a flattened weight so all E expert matmuls become ONE big GEMM:
        # We: (E, D, N) -> (D, E*N)
        We_flat = getattr(self, "_We_flat", None)
        if We_flat is None or We_flat.device != self.We.device:
            We_flat = self.We.permute(1, 0, 2).reshape(self.We.shape[1], -1).contiguous()
            self._We_flat = We_flat

        s = x.shape[0]
        n = self.We.shape[2]
        e = self.We.shape[0]

        # Routing gate (tiny GEMM + softmax)
        gate = torch.softmax(x @ self.Wr, dim=-1).contiguous()

        # Single fused GEMM for all experts: (S, D) @ (D, E*N) -> (S, E*N)
        outs = x @ We_flat

        # Fused weighted combine over experts + exact GELU in one Triton kernel
        y = torch.empty((s, n), device=x.device, dtype=x.dtype)
        BLOCK = 256
        grid = (s, triton.cdiv(n, BLOCK))
        _moe_combine_gelu_kernel[grid](
            gate, outs, y,
            N=n, E=e, BLOCK=BLOCK,
            num_warps=4,
        )
        return y

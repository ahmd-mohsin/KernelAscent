import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 400014
S, D, N, E, DT = 1024, 2048, 2048, 4, torch.float16


@triton.jit
def _combine_gelu_kernel(
    outs_ptr,      # (E, S, N) fp16, contiguous
    gate_ptr,      # (S, E) fp16, contiguous
    y_ptr,         # (S, N) fp16
    SN,            # S * N
    N,             # inner dim
    stride_e,      # S * N (expert stride in outs)
    E: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < SN
    s = offs // N

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for e in tl.static_range(E):
        g = tl.load(gate_ptr + s * E + e, mask=mask, other=0.0).to(tl.float32)
        o = tl.load(outs_ptr + e * stride_e + offs, mask=mask, other=0.0).to(tl.float32)
        acc += g * o

    # exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    y = acc * 0.5 * (1.0 + tl.math.erf(acc * 0.7071067811865476))
    tl.store(y_ptr + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wr = nn.Parameter((torch.randn(D, E, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.We = nn.Parameter((torch.randn(E, D, N, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        s, d = x.shape
        e, _, n = self.We.shape

        # gating (tiny matmul + softmax)
        gate = torch.softmax(x @ self.Wr, dim=-1).contiguous()  # (S, E) fp16

        # single batched GEMM for all experts: (E, S, N)
        outs = torch.matmul(x.unsqueeze(0), self.We)  # bmm under the hood
        outs = outs.contiguous()

        y = torch.empty((s, n), device=x.device, dtype=torch.float16)
        sn = s * n
        BLOCK = 1024
        grid = (triton.cdiv(sn, BLOCK),)
        _combine_gelu_kernel[grid](
            outs, gate, y,
            sn, n, sn,
            E=e, BLOCK=BLOCK,
            num_warps=4,
        )
        return y

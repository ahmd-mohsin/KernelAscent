import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 400009
S, D, N, E, DT = 1024, 1024, 1024, 8, torch.float16


@triton.jit
def _moe_mix_gelu_kernel(
    outs_ptr, gate_ptr, y_ptr,
    S, N,
    E: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs_n < N

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    base = pid_s * N + offs_n
    for e in tl.static_range(E):
        g = tl.load(gate_ptr + pid_s * E + e)  # fp16 scalar
        o = tl.load(outs_ptr + e * S * N + base, mask=mask, other=0.0)  # fp16
        p = g * o  # fp16 product (matches reference elementwise fp16 mul)
        acc += p.to(tl.float32)  # fp32 accumulation (matches torch reduction opmath)

    # cast to fp16 (sum output dtype), then GELU in fp32 opmath, store fp16
    yv = acc.to(tl.float16).to(tl.float32)
    inv_sqrt2 = 0.7071067811865476
    r = 0.5 * yv * (1.0 + tl.math.erf(yv * inv_sqrt2))
    tl.store(y_ptr + base, r.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wr = nn.Parameter((torch.randn(D, E, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.We = nn.Parameter((torch.randn(E, D, N, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            gate = torch.softmax(x @ self.Wr, dim=-1)
            outs = torch.stack([x @ self.We[e] for e in range(self.We.shape[0])], dim=0)
            y = (gate.transpose(0, 1).unsqueeze(-1) * outs).sum(0)
            return F.gelu(y)

        x = x.contiguous()
        s, d = x.shape
        e, _, n = self.We.shape

        # Gating: small matmul + softmax (softmax internally computed in fp32)
        gate = torch.softmax(x @ self.Wr, dim=-1).contiguous()  # (S, E) fp16

        # All expert projections in one batched matmul (broadcast x across experts)
        outs = torch.matmul(x, self.We)  # (E, S, N) fp16
        outs = outs.contiguous()

        y = torch.empty((s, n), device=x.device, dtype=torch.float16)

        BLOCK_N = 256
        grid = (s, triton.cdiv(n, BLOCK_N))
        _moe_mix_gelu_kernel[grid](
            outs, gate, y,
            s, n,
            E=e,
            BLOCK_N=BLOCK_N,
            num_warps=4,
        )
        return y

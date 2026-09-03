import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 400005
S, D, N, E, DT = 512, 2048, 1024, 8, torch.float16


@triton.jit
def _fused_moe_gelu_kernel(
    OUT_ptr,   # (S, E*N) fp16, expert outputs laid out [s, e, n]
    GATE_ptr,  # (S, E) fp16 softmax gates
    Y_ptr,     # (S, N) fp16 result
    N,         # runtime N
    E: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    e = tl.arange(0, E)
    # gates for this row (fp16)
    gate = tl.load(GATE_ptr + pid_s * E + e)

    # expert outputs block: (E, BLOCK_N), fp16
    ptrs = OUT_ptr + pid_s * (E * N) + e[:, None] * N + offs_n[None, :]
    outs = tl.load(ptrs, mask=mask_n[None, :], other=0.0)

    # elementwise product in fp16 (round once, matching torch's fp16 mul),
    # then reduce over experts with fp32 accumulation (matching torch.sum on half)
    p = gate[:, None] * outs
    acc = tl.sum(p.to(tl.float32), axis=0)

    # cast to fp16 (this is tensor `y` in the reference), then gelu computed in fp32
    y16 = acc.to(tl.float16)
    yf = y16.to(tl.float32)
    g = 0.5 * yf * (1.0 + tl.math.erf(yf * 0.7071067811865476))

    tl.store(Y_ptr + pid_s * N + offs_n, g.to(tl.float16), mask=mask_n)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wr = nn.Parameter((torch.randn(D, E, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.We = nn.Parameter((torch.randn(E, D, N, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self._Wflat = None

    def forward(self, x):
        if x.is_cuda:
            # lazily build fused expert weight matrix (D, E*N) so all expert
            # matmuls become one big GEMM
            if self._Wflat is None or self._Wflat.device != x.device:
                e_, d_, n_ = self.We.shape
                self._Wflat = self.We.permute(1, 0, 2).reshape(d_, e_ * n_).contiguous()

            e_, d_, n_ = self.We.shape
            s_ = x.shape[0]

            # gating (same numerics as reference)
            gate = torch.softmax(x @ self.Wr, dim=-1).contiguous()

            # all experts in a single GEMM: (S, D) @ (D, E*N) -> (S, E*N)
            outs = x @ self._Wflat

            y = torch.empty((s_, n_), device=x.device, dtype=x.dtype)

            BLOCK_N = 256
            grid = (s_, triton.cdiv(n_, BLOCK_N))
            _fused_moe_gelu_kernel[grid](
                outs, gate, y, n_,
                E=e_, BLOCK_N=BLOCK_N,
                num_warps=4,
            )
            return y

        # CPU fallback: reference implementation
        gate = torch.softmax(x @ self.Wr, dim=-1)
        outs = torch.stack([x @ self.We[e] for e in range(self.We.shape[0])], dim=0)
        y = (gate.transpose(0, 1).unsqueeze(-1) * outs).sum(0)
        return F.gelu(y)

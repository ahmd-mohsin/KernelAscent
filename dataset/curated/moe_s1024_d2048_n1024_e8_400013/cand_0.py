import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 400013
S, D, N, E, DT = 1024, 2048, 1024, 8, torch.float16


@triton.jit
def _moe_fuse_kernel(
    logits_ptr,   # (S, E) fp16 logits = x @ Wr
    outs_ptr,     # (S, E*N) fp16 = x @ We_flat
    y_ptr,        # (S, N) fp16 output
    Ncols,
    E: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_n = tl.program_id(1)

    # softmax over the E gate logits (fp32 for stability, matches torch)
    offs_e = tl.arange(0, E)
    lg = tl.load(logits_ptr + pid_s * E + offs_e).to(tl.float32)
    m = tl.max(lg, 0)
    p = tl.exp(lg - m)
    p = p / tl.sum(p, 0)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs_n < Ncols

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    base = outs_ptr + pid_s * E * Ncols
    for e in tl.static_range(E):
        v = tl.load(base + e * Ncols + offs_n, mask=mask, other=0.0).to(tl.float32)
        w = tl.sum(tl.where(offs_e == e, p, 0.0), 0)
        acc += v * w

    # exact GELU (erf variant, matching F.gelu default)
    g = acc * 0.5 * (1.0 + tl.math.erf(acc * 0.7071067811865476))
    tl.store(y_ptr + pid_s * Ncols + offs_n, g.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wr = nn.Parameter((torch.randn(D, E, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.We = nn.Parameter((torch.randn(E, D, N, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        e, d, n = self.We.shape

        if not x.is_cuda:
            gate = torch.softmax(x @ self.Wr, dim=-1)
            outs = torch.matmul(x.unsqueeze(0), self.We)
            y = (gate.transpose(0, 1).unsqueeze(-1) * outs).sum(0)
            return F.gelu(y)

        # cache flattened expert weights: (D, E*N) so all experts run in one GEMM
        Wf = getattr(self, "_We_flat", None)
        if Wf is None or Wf.device != x.device:
            Wf = self.We.permute(1, 0, 2).reshape(d, e * n).contiguous()
            self._We_flat = Wf

        x = x.contiguous()
        s = x.shape[0]

        logits = x @ self.Wr          # (S, E)
        outs = x @ Wf                 # (S, E*N) single large GEMM

        y = torch.empty((s, n), device=x.device, dtype=torch.float16)

        BLOCK_N = 1024
        grid = (s, triton.cdiv(n, BLOCK_N))
        _moe_fuse_kernel[grid](
            logits, outs, y, n,
            E=e, BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return y

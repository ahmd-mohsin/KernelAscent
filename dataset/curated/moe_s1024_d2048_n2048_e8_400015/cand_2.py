import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 400015
S, D, N, E, DT = 1024, 2048, 2048, 8, torch.float16


@triton.jit
def _moe_combine_gelu_kernel(
    logits_ptr,   # (S, E) fp16 gating logits
    outs_ptr,     # (S, E*N) fp16 expert outputs, layout s*E*N + e*N + n
    y_ptr,        # (S, N) fp16 output
    N_dim,        # N
    E_dim: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N_dim
    offs_e = tl.arange(0, E_dim)

    # softmax over the E gating logits (computed in fp32, stored as fp16 like torch)
    logits = tl.load(logits_ptr + pid_s * E_dim + offs_e).to(tl.float32)
    m = tl.max(logits, axis=0)
    p = tl.exp(logits - m)
    denom = tl.sum(p, axis=0)
    gate = (p / denom).to(tl.float16)  # (E,)

    # load expert outputs block: (E, BLOCK_N)
    ptrs = outs_ptr + pid_s * E_dim * N_dim + offs_e[:, None] * N_dim + offs_n[None, :]
    outs = tl.load(ptrs, mask=mask_n[None, :], other=0.0)  # fp16

    # fp16 elementwise product (matches reference), fp32 accumulation (matches torch.sum)
    prod = gate[:, None] * outs                 # fp16
    acc = tl.sum(prod.to(tl.float32), axis=0)   # fp32 accumulate
    y = acc.to(tl.float16)

    # exact (erf) GELU computed in fp32 as torch does for half inputs
    yf = y.to(tl.float32)
    g = 0.5 * yf * (1.0 + tl.math.erf(yf * 0.7071067811865476))
    tl.store(y_ptr + pid_s * N_dim + offs_n, g.to(tl.float16), mask=mask_n)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wr = nn.Parameter((torch.randn(D, E, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.We = nn.Parameter((torch.randn(E, D, N, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self._We_flat = None  # cached (D, E*N) layout for a single big GEMM

    def forward(self, x):
        if self._We_flat is None or self._We_flat.device != x.device:
            # (E, D, N) -> (D, E, N) -> (D, E*N), contiguous copy done once
            self._We_flat = self.We.permute(1, 0, 2).reshape(self.We.shape[1], -1).contiguous()

        Ecnt, Ddim, Ndim = self.We.shape
        Srows = x.shape[0]

        # gating logits: small GEMM (S, D) @ (D, E)
        logits = x @ self.Wr  # (S, E) fp16

        # all expert outputs in one large GEMM: (S, D) @ (D, E*N)
        outs = x @ self._We_flat  # (S, E*N) fp16, layout s*E*N + e*N + n

        y = torch.empty((Srows, Ndim), device=x.device, dtype=x.dtype)

        BLOCK_N = 256
        grid = (Srows, triton.cdiv(Ndim, BLOCK_N))
        _moe_combine_gelu_kernel[grid](
            logits, outs, y,
            Ndim,
            E_dim=Ecnt,
            BLOCK_N=BLOCK_N,
            num_warps=4,
        )
        return y

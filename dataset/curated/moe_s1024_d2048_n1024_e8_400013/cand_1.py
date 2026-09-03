import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 400013
S, D, N, E, DT = 1024, 2048, 1024, 8, torch.float16


@triton.jit
def _fused_gate_gelu_kernel(
    logits_ptr,      # (S, E) fp16 gating logits
    outs_ptr,        # (S, E*N) fp16 expert outputs
    y_ptr,           # (S, N) fp16 output
    N,
    stride_ls,
    stride_os,
    stride_oe,
    stride_ys,
    E: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    s = tl.program_id(0)
    nb = tl.program_id(1)

    e = tl.arange(0, E)
    # softmax over the E gating logits (computed in fp32, rounded to fp16 like torch)
    logits = tl.load(logits_ptr + s * stride_ls + e).to(tl.float32)
    m = tl.max(logits, axis=0)
    p = tl.exp(logits - m)
    gate = (p / tl.sum(p, axis=0)).to(tl.float16)

    offs = nb * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N

    ptrs = outs_ptr + s * stride_os + e[:, None] * stride_oe + offs[None, :]
    o = tl.load(ptrs, mask=mask[None, :], other=0.0)  # fp16 (E, BLOCK_N)

    # elementwise product in fp16 (matches reference), accumulate in fp32 (matches torch sum)
    prod = gate[:, None] * o
    acc = tl.sum(prod.to(tl.float32), axis=0)

    # exact GELU (erf form), computed in fp32 like torch's half gelu
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    y = 0.5 * acc * (1.0 + tl.math.erf(acc * INV_SQRT2))

    tl.store(y_ptr + s * stride_ys + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wr = nn.Parameter((torch.randn(D, E, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.We = nn.Parameter((torch.randn(E, D, N, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self._We_flat = None

    def forward(self, x):
        if self._We_flat is None or self._We_flat.device != x.device:
            # (E, D, N) -> (D, E*N) so all experts run in one big GEMM
            self._We_flat = self.We.permute(1, 0, 2).reshape(self.We.shape[1], -1).contiguous()

        x = x.contiguous()
        s, d = x.shape
        e = self.We.shape[0]
        n = self.We.shape[2]

        # single small GEMM for gating logits
        logits = x @ self.Wr                      # (S, E)
        # single large GEMM for all experts (tensor cores, fp16)
        outs = x @ self._We_flat                  # (S, E*N)

        y = torch.empty((s, n), device=x.device, dtype=x.dtype)

        BLOCK_N = 256
        grid = (s, triton.cdiv(n, BLOCK_N))
        _fused_gate_gelu_kernel[grid](
            logits, outs, y,
            n,
            logits.stride(0),
            outs.stride(0), n,
            y.stride(0),
            E=e,
            BLOCK_N=BLOCK_N,
            num_warps=4,
        )
        return y

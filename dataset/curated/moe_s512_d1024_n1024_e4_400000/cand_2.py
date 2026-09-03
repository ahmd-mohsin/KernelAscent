import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 400000
S, D, N, E, DT = 512, 1024, 1024, 4, torch.float16


@triton.jit
def _fused_gate_sum_gelu(
    logits_ptr,   # (S, E) fp16
    outs_ptr,     # (S, E*N) fp16, laid out [row, e, n]
    y_ptr,        # (S, N) fp16
    N: tl.constexpr,
    E: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cb = tl.program_id(1)

    # --- softmax over E logits (fp32 math, round to fp16 like reference gate) ---
    e_offs = tl.arange(0, E)
    lg = tl.load(logits_ptr + row * E + e_offs).to(tl.float32)
    m = tl.max(lg, axis=0)
    ex = tl.exp(lg - m)
    p = ex / tl.sum(ex, axis=0)
    p = p.to(tl.float16).to(tl.float32)  # match reference: gate stored in fp16

    # --- weighted sum over experts ---
    col = cb * BLOCK + tl.arange(0, BLOCK)
    mask = col < N
    ptrs = outs_ptr + row * (E * N) + e_offs[:, None] * N + col[None, :]
    vals = tl.load(ptrs, mask=mask[None, :], other=0.0).to(tl.float32)  # (E, BLOCK)
    acc = tl.sum(p[:, None] * vals, axis=0)  # (BLOCK,)

    # --- exact GELU (erf) ---
    y = 0.5 * acc * (1.0 + tl.math.erf(acc * 0.7071067811865476))

    tl.store(y_ptr + row * N + col, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wr = nn.Parameter((torch.randn(D, E, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.We = nn.Parameter((torch.randn(E, D, N, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self._We_cat = None  # (D, E*N) cached

    def forward(self, x):
        if self._We_cat is None or self._We_cat.device != x.device:
            # (E, D, N) -> (D, E, N) -> (D, E*N): x @ We_cat[:, e*N:(e+1)*N] == x @ We[e]
            self._We_cat = self.We.permute(1, 0, 2).reshape(self.We.shape[1], -1).contiguous()

        s = x.shape[0]
        logits = x @ self.Wr                      # (S, E)
        outs = x @ self._We_cat                   # (S, E*N) single big GEMM
        y = torch.empty((s, N), device=x.device, dtype=x.dtype)

        BLOCK = 256
        grid = (s, triton.cdiv(N, BLOCK))
        _fused_gate_sum_gelu[grid](
            logits, outs, y,
            N=N, E=E, BLOCK=BLOCK,
            num_warps=4,
        )
        return y

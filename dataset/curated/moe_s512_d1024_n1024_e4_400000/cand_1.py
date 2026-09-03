import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 400000
S, D, N, E, DT = 512, 1024, 1024, 4, torch.float16


@triton.jit
def _moe_softmax_wsum_gelu_kernel(
    buf_ptr,          # (S, E + E*N) fp16: [:, :E] logits, [:, E:] expert outputs
    y_ptr,            # (S, N) fp16 output
    N,
    stride_b, stride_y,
    E: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)

    # ---- softmax over the E gate logits (fp32 accumulation, like torch half softmax) ----
    e = tl.arange(0, E)
    lg = tl.load(buf_ptr + row * stride_b + e).to(tl.float32)
    m = tl.max(lg, 0)
    p = tl.exp(lg - m)
    p = p / tl.sum(p, 0)
    # match reference: gate is materialized in fp16 before the weighted sum
    p = p.to(tl.float16).to(tl.float32)

    # ---- weighted sum of the E expert outputs ----
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    ptrs = buf_ptr + row * stride_b + E + e[:, None] * N + offs[None, :]
    v = tl.load(ptrs, mask=mask[None, :], other=0.0).to(tl.float32)
    acc = tl.sum(p[:, None] * v, 0)

    # ---- exact GELU (erf-based, same as F.gelu default) ----
    r = 0.5 * acc * (1.0 + tl.math.erf(acc * 0.7071067811865476))

    tl.store(y_ptr + row * stride_y + offs, r.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wr = nn.Parameter((torch.randn(D, E, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.We = nn.Parameter((torch.randn(E, D, N, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def _get_wcat(self, x):
        wc = getattr(self, "_Wcat", None)
        if wc is None or wc.device != x.device or wc.dtype != x.dtype:
            # Fuse gate projection and all E expert matmuls into ONE GEMM:
            # columns [0:E] -> Wr, columns [E : E + E*N] -> We[e] laid out contiguously per expert
            Wflat = self.We.permute(1, 0, 2).reshape(D, E * N)  # (D, E*N)
            wc = torch.cat([self.Wr.to(x.dtype), Wflat.to(x.dtype)], dim=1).contiguous()
            self._Wcat = wc
        return wc

    def forward(self, x):
        if not x.is_cuda:
            gate = torch.softmax(x @ self.Wr, dim=-1)
            outs = torch.stack([x @ self.We[e] for e in range(E)], dim=0)
            y = (gate.transpose(0, 1).unsqueeze(-1) * outs).sum(0)
            return F.gelu(y)

        Wcat = self._get_wcat(x)

        # Single tensor-core GEMM producing gate logits + all expert outputs
        buf = x @ Wcat  # (S, E + E*N)

        s = x.shape[0]
        y = torch.empty((s, N), device=x.device, dtype=x.dtype)

        BLOCK = triton.next_power_of_2(N)
        _moe_softmax_wsum_gelu_kernel[(s,)](
            buf, y,
            N,
            buf.stride(0), y.stride(0),
            E=E,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y

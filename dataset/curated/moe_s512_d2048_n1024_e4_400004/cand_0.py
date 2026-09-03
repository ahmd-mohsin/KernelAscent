import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 400004
S, D, N, E, DT = 512, 2048, 1024, 4, torch.float16


@triton.jit
def _gate_mix_gelu_kernel(
    gate_ptr,        # (S, E) fp16
    big_ptr,         # (S, E*N) fp16, layout: [.., e*N + n]
    out_ptr,         # (S, N) fp16
    N,
    E: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs = pid_n * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    acc = tl.zeros((BLOCK,), dtype=tl.float16)
    row_base = big_ptr + pid_s * (E * N)
    for e in tl.static_range(E):
        g = tl.load(gate_ptr + pid_s * E + e)                       # fp16 scalar
        v = tl.load(row_base + e * N + offs, mask=mask, other=0.0)  # fp16
        acc = acc + g * v                                           # fp16 mul/add (matches ref)

    xf = acc.to(tl.float32)
    # exact (erf-based) GELU, computed in fp32 like torch's fp16 GELU
    y = 0.5 * xf * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    tl.store(out_ptr + pid_s * N + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wr = nn.Parameter((torch.randn(D, E, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.We = nn.Parameter((torch.randn(E, D, N, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def _get_wflat(self):
        # Flatten (E, D, N) -> (D, E*N) so all expert GEMMs run as ONE big GEMM.
        # Column blocks are independent, so results are bit-identical to per-expert GEMMs.
        wf = getattr(self, "_Wflat", None)
        if wf is None or wf.device != self.We.device or wf.dtype != self.We.dtype:
            wf = self.We.permute(1, 0, 2).reshape(self.We.shape[1], -1).contiguous()
            self._Wflat = wf
        return wf

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback (reference path)
            gate = torch.softmax(x @ self.Wr, dim=-1)
            outs = torch.stack([x @ self.We[e] for e in range(self.We.shape[0])], dim=0)
            y = (gate.transpose(0, 1).unsqueeze(-1) * outs).sum(0)
            return F.gelu(y)

        Wflat = self._get_wflat()
        E_ = self.We.shape[0]
        N_ = self.We.shape[2]
        S_ = x.shape[0]

        # small GEMM + softmax for the gate (same op as reference)
        gate = torch.softmax(x @ self.Wr, dim=-1).contiguous()  # (S, E) fp16

        # single fused GEMM for all experts
        big = x @ Wflat  # (S, E*N) fp16

        out = torch.empty((S_, N_), device=x.device, dtype=x.dtype)

        BLOCK = 1024
        grid = (S_, triton.cdiv(N_, BLOCK))
        _gate_mix_gelu_kernel[grid](
            gate, big, out, N_,
            E=E_, BLOCK=BLOCK,
            num_warps=8,
        )
        return out

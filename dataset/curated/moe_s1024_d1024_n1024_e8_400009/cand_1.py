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
    outs_ptr,   # (S, E*N) fp16, contiguous
    gate_ptr,   # (S, E)   fp16, contiguous
    y_ptr,      # (S, N)   fp16, contiguous
    Ncols,
    Ecount: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cb = tl.program_id(1)
    cols = cb * BLOCK + tl.arange(0, BLOCK)
    mask = cols < Ncols

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    base = row * Ecount * Ncols
    for e in tl.static_range(Ecount):
        g = tl.load(gate_ptr + row * Ecount + e).to(tl.float32)
        v = tl.load(outs_ptr + base + e * Ncols + cols, mask=mask, other=0.0).to(tl.float32)
        acc += g * v

    # exact (erf-based) GELU, computed in fp32 like PyTorch's half opmath
    r = 0.5 * acc * (1.0 + tl.math.erf(acc * 0.7071067811865476))

    tl.store(y_ptr + row * Ncols + cols, r.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wr = nn.Parameter((torch.randn(D, E, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.We = nn.Parameter((torch.randn(E, D, N, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # reference fallback for CPU
            gate = torch.softmax(x @ self.Wr, dim=-1)
            outs = torch.stack([x @ self.We[e] for e in range(self.We.shape[0])], dim=0)
            y = (gate.transpose(0, 1).unsqueeze(-1) * outs).sum(0)
            return F.gelu(y)

        # Lazily cache a (D, E*N) flattened weight so all expert GEMMs
        # become one large GEMM (much better A100 tensor-core utilization).
        Wflat = getattr(self, "_Wflat", None)
        if Wflat is None or Wflat.device != x.device:
            Ecount, Dsz, Ncols = self.We.shape
            Wflat = self.We.permute(1, 0, 2).reshape(Dsz, Ecount * Ncols).contiguous()
            self._Wflat = Wflat

        Ecount = self.We.shape[0]
        Ncols = self.We.shape[2]

        # gating: (S, E) — tiny GEMM + softmax
        gate = torch.softmax(x @ self.Wr, dim=-1).contiguous()

        # all experts in one GEMM: (S, D) @ (D, E*N) -> (S, E*N)
        outs = (x @ Wflat).contiguous()

        Ssz = x.shape[0]
        y = torch.empty((Ssz, Ncols), device=x.device, dtype=x.dtype)

        BLOCK = 1024
        grid = (Ssz, triton.cdiv(Ncols, BLOCK))
        _moe_mix_gelu_kernel[grid](
            outs, gate, y, Ncols,
            Ecount=Ecount, BLOCK=BLOCK,
            num_warps=8,
        )
        return y

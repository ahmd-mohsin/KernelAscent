import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 400002
S, D, N, E, DT = 512, 1024, 2048, 4, torch.float16


@triton.jit
def _combine_gelu_kernel(
    out_ptr,      # (S, E, N) fp16 - expert outputs
    gate_ptr,     # (S, E) fp16 - softmax gates
    y_ptr,        # (S, N) fp16 - result
    Ncols,
    E: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cb = tl.program_id(1)
    cols = cb * BLOCK + tl.arange(0, BLOCK)
    mask = cols < Ncols

    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for e in tl.static_range(E):
        g = tl.load(gate_ptr + row * E + e).to(tl.float32)
        v = tl.load(out_ptr + (row * E + e) * Ncols + cols, mask=mask, other=0.0).to(tl.float32)
        acc += g * v

    # exact GELU (erf-based), matching F.gelu default
    r = 0.5 * acc * (1.0 + tl.math.erf(acc * 0.7071067811865476))
    tl.store(y_ptr + row * Ncols + cols, r.to(y_ptr.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wr = nn.Parameter((torch.randn(D, E, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.We = nn.Parameter((torch.randn(E, D, N, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self._We_flat = None  # lazy cache: (D, E*N) contiguous fused expert weight

    def forward(self, x):
        We = self.We
        if (self._We_flat is None
                or self._We_flat.device != We.device
                or self._We_flat.dtype != We.dtype):
            # (E, D, N) -> (D, E, N) -> (D, E*N): one big GEMM instead of E small ones
            self._We_flat = We.permute(1, 0, 2).reshape(We.shape[1], -1).contiguous()

        Ecnt, Dm, Ncols = We.shape[0], We.shape[1], We.shape[2]
        Srows = x.shape[0]

        # gate: tiny GEMM + softmax (identical math to reference)
        gate = torch.softmax(x @ self.Wr, dim=-1).contiguous()  # (S, E)

        # all experts in one GEMM: (S, D) @ (D, E*N) -> (S, E*N) == (S, E, N)
        outs = x @ self._We_flat  # (S, E*N), row-major so [s, e, n] laid out contiguously

        y = torch.empty((Srows, Ncols), device=x.device, dtype=x.dtype)

        BLOCK = 1024
        grid = (Srows, triton.cdiv(Ncols, BLOCK))
        _combine_gelu_kernel[grid](
            outs, gate, y, Ncols,
            E=Ecnt, BLOCK=BLOCK,
            num_warps=8,
        )
        return y

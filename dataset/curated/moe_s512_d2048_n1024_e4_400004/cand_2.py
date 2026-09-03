import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 400004
S, D, N, E, DT = 512, 2048, 1024, 4, torch.float16


@triton.jit
def _fused_moe_gelu_kernel(
    out_ptr,      # (S, E, N) fp16, contiguous
    gate_ptr,     # (S, E) fp16/fp32
    y_ptr,        # (S, N) fp16
    N,            # int
    E: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    nb = tl.program_id(1)
    cols = nb * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N

    acc = tl.zeros([BLOCK], dtype=tl.float32)
    base = row * E * N
    for e in tl.static_range(E):
        g = tl.load(gate_ptr + row * E + e).to(tl.float32)
        o = tl.load(out_ptr + base + e * N + cols, mask=mask, other=0.0).to(tl.float32)
        acc += g * o

    # exact (erf-based) GELU
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    y = 0.5 * acc * (1.0 + tl.math.erf(acc * INV_SQRT2))

    tl.store(y_ptr + row * N + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wr = nn.Parameter((torch.randn(D, E, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.We = nn.Parameter((torch.randn(E, D, N, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        We = self.We
        E_, D_, N_ = We.shape
        S_ = x.shape[0]

        # Cache a (D, E*N) flattened expert weight so all expert matmuls
        # become a single large GEMM.
        We_flat = getattr(self, "_We_flat", None)
        if (
            We_flat is None
            or We_flat.device != We.device
            or We_flat.dtype != We.dtype
            or getattr(self, "_We_version", None) != We._version
        ):
            We_flat = We.permute(1, 0, 2).reshape(D_, E_ * N_).contiguous()
            self._We_flat = We_flat
            self._We_version = We._version

        # Router gate
        gate = torch.softmax(x @ self.Wr, dim=-1).contiguous()  # (S, E)

        # All experts in one GEMM: (S, D) @ (D, E*N) -> (S, E, N)
        outs = (x @ We_flat).view(S_, E_, N_)

        if x.is_cuda:
            y = torch.empty((S_, N_), device=x.device, dtype=torch.float16)
            BLOCK = 256
            grid = (S_, triton.cdiv(N_, BLOCK))
            _fused_moe_gelu_kernel[grid](
                outs, gate, y, N_, E=E_, BLOCK=BLOCK, num_warps=4
            )
            return y
        else:
            y = (gate.unsqueeze(-1) * outs).sum(1)
            return F.gelu(y)

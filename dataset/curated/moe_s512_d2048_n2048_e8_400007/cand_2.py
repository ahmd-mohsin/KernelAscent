import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 400007
S, D, N, E, DT = 512, 2048, 2048, 8, torch.float16


@triton.jit
def _mix_gelu_kernel(
    Z_ptr,        # (S, E, N) fp16 : expert outputs
    G_ptr,        # (S, E)    fp16 : gate values
    Y_ptr,        # (S, N)    fp16 : output
    total,        # S * N
    N_dim,        # N
    E_dim: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    s = offs // N_dim
    n = offs % N_dim

    acc = tl.zeros([BLOCK], dtype=tl.float32)
    base_z = s * (E_dim * N_dim) + n
    base_g = s * E_dim
    for e in tl.static_range(E_dim):
        z = tl.load(Z_ptr + base_z + e * N_dim, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(G_ptr + base_g + e, mask=mask, other=0.0).to(tl.float32)
        # match reference: elementwise product rounded to fp16, then fp32 accumulate
        p = (g * z).to(tl.float16).to(tl.float32)
        acc += p

    # match reference: sum cast to fp16, gelu computed in fp32 (erf-based), stored fp16
    y = acc.to(tl.float16).to(tl.float32)
    r = 0.5 * y * (1.0 + tl.math.erf(y * 0.7071067811865476))
    tl.store(Y_ptr + offs, r.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wr = nn.Parameter((torch.randn(D, E, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.We = nn.Parameter((torch.randn(E, D, N, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self._We_flat = None  # cached (D, E*N) layout for a single big GEMM

    def forward(self, x):
        Wr = self.Wr
        We = self.We
        if x.is_cuda:
            if self._We_flat is None or self._We_flat.device != x.device:
                # (E, D, N) -> (D, E, N) -> (D, E*N), so column blocks match per-expert GEMMs
                self._We_flat = We.permute(1, 0, 2).reshape(We.shape[1], -1).contiguous()

            gate = torch.softmax(x @ Wr, dim=-1).contiguous()  # (S, E) fp16

            # One large tensor-core GEMM replacing E separate GEMMs
            Z = (x @ self._We_flat)  # (S, E*N) fp16, contiguous
            Sd, Ed = gate.shape
            Nd = Z.shape[1] // Ed
            y = torch.empty((Sd, Nd), device=x.device, dtype=torch.float16)

            total = Sd * Nd
            BLOCK = 1024
            grid = (triton.cdiv(total, BLOCK),)
            _mix_gelu_kernel[grid](Z, gate, y, total, Nd, E_dim=Ed, BLOCK=BLOCK)
            return y
        else:
            gate = torch.softmax(x @ Wr, dim=-1)
            outs = torch.stack([x @ We[e] for e in range(We.shape[0])], dim=0)
            y = (gate.transpose(0, 1).unsqueeze(-1) * outs).sum(0)
            return F.gelu(y)

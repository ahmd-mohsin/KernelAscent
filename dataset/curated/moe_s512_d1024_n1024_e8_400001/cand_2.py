import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 400001
S, D, N, E, DT = 512, 1024, 1024, 8, torch.float16


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wr = nn.Parameter((torch.randn(D, E, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.We = nn.Parameter((torch.randn(E, D, N, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        # x: (S, D)
        S_, D_ = x.shape
        E_, _, N_ = self.We.shape

        # gate: (S, E) -- small GEMM + softmax (softmax internally accumulates in fp32)
        gate = torch.softmax(x @ self.Wr, dim=-1)

        # Fold the gate into the input per-expert:
        #   y[s, n] = sum_e gate[s, e] * (x @ We[e])[s, n]
        #           = sum_e sum_d (gate[s, e] * x[s, d]) * We[e, d, n]
        # This collapses E separate GEMMs + weighted reduction into ONE big GEMM:
        #   (S, E*D) @ (E*D, N)
        # gx: (S, E, D) broadcasted multiply, then flatten
        gx = (gate.unsqueeze(-1) * x.unsqueeze(1)).reshape(S_, E_ * D_)

        # We is contiguous (E, D, N) so this reshape is a free view with the
        # correct (e*D + d, n) layout.
        We_flat = self.We.reshape(E_ * D_, N_)

        # Single large GEMM with fp32 accumulation inside cuBLAS/tensor cores
        y = gx @ We_flat

        # Fused exact (erf) GELU
        return F.gelu(y)

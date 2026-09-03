import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 400003
S, D, N, E, DT = 512, 1024, 2048, 8, torch.float16


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wr = nn.Parameter((torch.randn(D, E, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.We = nn.Parameter((torch.randn(E, D, N, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        # gate: (S, E) -- tiny GEMM + softmax (softmax accumulates in fp32 internally)
        gate = torch.softmax(x @ self.Wr, dim=-1)

        Eh, Dh, Nh = self.We.shape
        Sh = x.shape[0]

        # Key optimization: move the gating in front of the matmul.
        #   y[s] = sum_e gate[s,e] * (x[s] @ We[e]) = concat_e(gate[s,e] * x[s]) @ concat_e(We[e])
        # This collapses E separate GEMMs + a weighted-sum reduction into ONE large
        # tensor-core GEMM of shape (S, E*D) @ (E*D, N), with fp32 accumulation
        # inside cuBLAS (replacing the fp16 stack/multiply/sum epilogue).
        xs = (gate.unsqueeze(-1) * x.unsqueeze(1)).reshape(Sh, Eh * Dh)   # (S, E*D)
        W = self.We.reshape(Eh * Dh, Nh)                                  # zero-copy view (S contiguous)

        y = xs @ W                                                        # single big GEMM
        return F.gelu(y)

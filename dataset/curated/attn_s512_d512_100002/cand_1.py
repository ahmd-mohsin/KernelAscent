import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 100002
S, D, DT = 512, 512, torch.bfloat16

class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self._Wqkv = None

    def forward(self, x):
        # Lazily build fused QKV weight on the correct device (one GEMM instead of three)
        if (self._Wqkv is None
                or self._Wqkv.device != x.device
                or self._Wqkv.dtype != self.Wq.dtype):
            self._Wqkv = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).contiguous()

        d = self.Wq.shape[0]
        qkv = x @ self._Wqkv  # (S, 3D) single GEMM
        q, k, v = qkv[:, :d], qkv[:, d:2 * d], qkv[:, 2 * d:]

        # Fused (flash) attention: softmax(q k^T / sqrt(d)) v in one kernel
        out = F.scaled_dot_product_attention(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        )
        return out.squeeze(0)

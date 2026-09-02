import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 100020
S, D, DT = 1024, 2048, torch.float16


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def _get_fused_weight(self, device, dtype):
        w = getattr(self, "_Wqkv", None)
        if w is None or w.device != device or w.dtype != dtype:
            # Single fused projection weight: one big GEMM instead of three.
            w = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).to(device=device, dtype=dtype).contiguous()
            self._Wqkv = w
        return w

    def forward(self, x):
        d = self.Wq.shape[0]
        Wqkv = self._get_fused_weight(x.device, x.dtype)

        # Fused QKV projection: (S, D) @ (D, 3D) -> (S, 3D), one tensor-core GEMM.
        qkv = x @ Wqkv
        q, k, v = qkv[:, :d], qkv[:, d:2 * d], qkv[:, 2 * d:]

        scale = 1.0 / math.sqrt(d)
        # Scores GEMM with fused scaling (baddbmm-style via matmul + mul is cheap;
        # torch.softmax is a single fused CUDA kernel).
        scores = torch.matmul(q, k.transpose(-1, -2))
        scores.mul_(scale)
        a = torch.softmax(scores, dim=-1)
        return a @ v


def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(S, D, generator=g).to(DT)]

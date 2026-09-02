import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 100013
S, D, DT = 1024, 512, torch.float16


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def _get_fused_w(self, device, dtype):
        w = getattr(self, "_Wqkv", None)
        if w is None or w.device != device or w.dtype != dtype:
            w = torch.cat((self.Wq, self.Wk, self.Wv), dim=1).to(device=device, dtype=dtype).contiguous()
            self._Wqkv = w
        return w

    def forward(self, x):
        d = x.shape[-1]
        w = self._get_fused_w(x.device, x.dtype)

        # Single fused GEMM for Q, K, V projections
        qkv = x @ w  # (S, 3D)
        q, k, v = qkv[:, :d], qkv[:, d:2 * d], qkv[:, 2 * d:]

        # Fused causal attention (flash / mem-efficient kernel), identical math:
        # softmax(QK^T / sqrt(d) + causal_mask) @ V
        q = q.unsqueeze(0).unsqueeze(0).contiguous()
        k = k.unsqueeze(0).unsqueeze(0).contiguous()
        v = v.unsqueeze(0).unsqueeze(0).contiguous()

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return out.squeeze(0).squeeze(0)

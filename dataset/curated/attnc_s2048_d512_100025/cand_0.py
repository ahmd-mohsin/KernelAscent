import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 100025
S, D, DT = 2048, 512, torch.float16


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def _get_fused_w(self, x):
        w = getattr(self, "_Wqkv", None)
        if w is None or w.device != self.Wq.device or w.dtype != self.Wq.dtype:
            w = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).contiguous()
            self._Wqkv = w
        return w

    def forward(self, x):
        d = self.Wq.shape[1]

        # Fused QKV projection: one GEMM instead of three
        qkv = x @ self._get_fused_w(x)
        q, k, v = qkv[..., :d], qkv[..., d:2 * d], qkv[..., 2 * d:]

        if x.is_cuda:
            # Reshape to (B, H, S, D) for SDPA; fused causal attention kernel
            q4 = q.unsqueeze(0).unsqueeze(0)
            k4 = k.unsqueeze(0).unsqueeze(0)
            v4 = v.unsqueeze(0).unsqueeze(0)
            try:
                out = F.scaled_dot_product_attention(
                    q4, k4, v4,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=True,
                    scale=1.0 / math.sqrt(d),
                )
                return out.squeeze(0).squeeze(0)
            except Exception:
                pass  # fall through to manual path

        # Fallback: manual causal attention (matches reference semantics)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(d)
        n = scores.shape[-1]
        causal_mask = torch.ones(n, n, dtype=torch.bool, device=scores.device).triu(diagonal=1)
        scores = scores.masked_fill(causal_mask, float('-inf'))
        a = torch.softmax(scores, dim=-1)
        return a @ v

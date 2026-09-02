import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 100000
S, D, DT = 512, 512, torch.float16


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        # Fuse the three projection GEMMs into a single GEMM by caching a
        # concatenated weight matrix (built lazily on first call / device move).
        Wqkv = getattr(self, "_Wqkv", None)
        if Wqkv is None or Wqkv.device != x.device or Wqkv.dtype != x.dtype:
            Wqkv = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).to(device=x.device, dtype=x.dtype).contiguous()
            self._Wqkv = Wqkv

        d = self.Wq.shape[0]
        qkv = x @ Wqkv  # (S, 3D) single large GEMM -> better tensor-core utilization
        q = qkv[:, :d]
        k = qkv[:, d:2 * d]
        v = qkv[:, 2 * d:]

        if x.is_cuda:
            # Fused attention (scale + QK^T + softmax + @V) in one kernel,
            # avoiding materialization of the full attention matrix in HBM.
            o = F.scaled_dot_product_attention(
                q.unsqueeze(0).unsqueeze(0),
                k.unsqueeze(0).unsqueeze(0),
                v.unsqueeze(0).unsqueeze(0),
                scale=1.0 / math.sqrt(d),
            )
            return o.squeeze(0).squeeze(0)
        else:
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(d)
            a = torch.softmax(scores, dim=-1)
            return a @ v

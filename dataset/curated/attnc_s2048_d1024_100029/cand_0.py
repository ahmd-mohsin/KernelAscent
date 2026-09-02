import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 100029
S, D, DT = 2048, 1024, torch.float16


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        # Lazily build (and cache) a fused QKV projection matrix so that the three
        # input GEMMs collapse into a single large GEMM (better tensor-core
        # utilization / fewer kernel launches on A100).
        Wqkv = getattr(self, "_Wqkv_cache", None)
        if (
            Wqkv is None
            or Wqkv.device != x.device
            or Wqkv.dtype != self.Wq.dtype
        ):
            Wqkv = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).contiguous()
            self._Wqkv_cache = Wqkv

        s, d = x.shape[-2], self.Wq.shape[0]

        # Single fused projection GEMM: (S, D) @ (D, 3D) -> (S, 3D)
        qkv = x @ Wqkv
        q, k, v = qkv.split(d, dim=-1)

        # Reshape to (batch=1, heads=1, seq, dim) for the fused attention kernel.
        q = q.unsqueeze(0).unsqueeze(0)
        k = k.unsqueeze(0).unsqueeze(0)
        v = v.unsqueeze(0).unsqueeze(0)

        # Fused (memory-efficient) causal attention: avoids materializing the
        # full S x S score matrix, the explicit triu mask, and separate softmax
        # kernels. Scale = 1/sqrt(head_dim) matches the reference exactly.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        return out.reshape(s, d)


def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(S, D, generator=g).to(DT)]

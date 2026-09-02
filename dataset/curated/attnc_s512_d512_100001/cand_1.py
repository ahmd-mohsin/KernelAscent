import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 100001
S, D, DT = 512, 512, torch.float16


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        # Lazily build & cache a fused QKV weight so all three projections
        # run as a single GEMM (better tensor-core utilization on A100).
        W = self.__dict__.get('_Wqkv', None)
        if W is None or W.device != x.device or W.dtype != x.dtype:
            W = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).contiguous().to(x.device, x.dtype)
            self.__dict__['_Wqkv'] = W

        d = self.Wq.shape[1]
        qkv = x @ W  # (S, 3D) single fused GEMM
        q = qkv[:, :d]
        k = qkv[:, d:2 * d]
        v = qkv[:, 2 * d:]

        # Fused causal attention (softmax(QK^T/sqrt(d)) @ V) via optimized
        # SDPA kernels — avoids materializing the SxS score matrix path
        # of separate matmul + mask + softmax + matmul ops.
        out = F.scaled_dot_product_attention(
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            is_causal=True,
            scale=1.0 / math.sqrt(d),
        )
        return out.squeeze(0)


def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(S, D, generator=g).to(DT)]

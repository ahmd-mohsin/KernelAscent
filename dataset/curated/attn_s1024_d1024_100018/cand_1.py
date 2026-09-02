import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 100018
S, D, DT = 1024, 1024, torch.bfloat16

class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self._Wqkv = None

    def _get_fused_weight(self):
        w = self._Wqkv
        if (
            w is None
            or w.device != self.Wq.device
            or w.dtype != self.Wq.dtype
        ):
            # Fuse the three projection matrices into a single [D, 3D] matrix so
            # Q, K, V are produced by ONE large GEMM (better tensor-core utilization
            # and 1/3 the kernel launches / activations reads of three GEMMs).
            w = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).contiguous()
            self._Wqkv = w
        return w

    def forward(self, x):
        d = x.shape[-1]
        Wqkv = self._get_fused_weight()

        # Single fused GEMM for Q, K, V.
        qkv = x @ Wqkv
        q, k, v = qkv.chunk(3, dim=-1)

        # FlashAttention kernel: fuses (q @ k^T) * scale, softmax, and (a @ v)
        # into one memory-efficient kernel — never materializes the S x S score
        # matrix in HBM. Accumulation is done in fp32 internally, matching the
        # reference's fp32 softmax accumulation to bf16 tolerance.
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        out = F.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(d))
        return out.squeeze(0)

def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(S, D, generator=g).to(DT)]

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100010
S, D, DT = 512, 2048, torch.bfloat16


@triton.jit
def _scaled_softmax_kernel(
    X, Y,
    N,
    stride_xm, stride_ym,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + row * stride_xm + offs, mask=mask, other=float('-inf')).to(tl.float32)
    # mimic reference: bf16 tensor / scalar -> fp32 opmath, rounded back to bf16
    x = (x / SCALE).to(tl.bfloat16).to(tl.float32)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(Y + row * stride_ym + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        d = x.shape[-1]

        # Lazily build a fused QKV weight so all three projections run as one GEMM.
        Wqkv = getattr(self, "_Wqkv", None)
        if (
            Wqkv is None
            or Wqkv.device != x.device
            or Wqkv.dtype != x.dtype
            or getattr(self, "_Wqkv_version", None) != (self.Wq._version, self.Wk._version, self.Wv._version)
        ):
            Wqkv = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).to(device=x.device, dtype=x.dtype).contiguous()
            self._Wqkv = Wqkv
            self._Wqkv_version = (self.Wq._version, self.Wk._version, self.Wv._version)

        orig_shape = x.shape
        x2 = x.reshape(-1, d)
        n = x2.shape[0]

        # Single fused GEMM for Q, K, V
        qkv = x2 @ Wqkv
        q = qkv[:, :d]
        k = qkv[:, d:2 * d]
        v = qkv[:, 2 * d:]

        # Attention scores (unscaled); scaling fused into softmax kernel
        scores = q @ k.transpose(-1, -2)

        if scores.is_cuda:
            a = torch.empty_like(scores)
            N = scores.shape[-1]
            BLOCK = triton.next_power_of_2(N)
            num_warps = 4 if BLOCK <= 1024 else 8
            _scaled_softmax_kernel[(scores.shape[0],)](
                scores, a,
                N,
                scores.stride(0), a.stride(0),
                SCALE=math.sqrt(d),
                BLOCK=BLOCK,
                num_warps=num_warps,
            )
        else:
            a = torch.softmax(scores / math.sqrt(d), dim=-1)

        out = a @ v
        return out.reshape(orig_shape[:-1] + (d,))

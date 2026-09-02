import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 200003
S, D, DT = 512, 512, torch.bfloat16


@triton.jit
def _rope_qk_kernel(
    Q, K, COS, SIN,
    half,
    stride_q, stride_k,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < half

    c = tl.load(COS + row * half + offs, mask=mask, other=0.0)
    s = tl.load(SIN + row * half + offs, mask=mask, other=0.0)

    q_ptr = Q + row * stride_q
    q1 = tl.load(q_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    q2 = tl.load(q_ptr + half + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(q_ptr + offs, (q1 * c - q2 * s).to(Q.dtype.element_ty), mask=mask)
    tl.store(q_ptr + half + offs, (q1 * s + q2 * c).to(Q.dtype.element_ty), mask=mask)

    k_ptr = K + row * stride_k
    k1 = tl.load(k_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    k2 = tl.load(k_ptr + half + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(k_ptr + offs, (k1 * c - k2 * s).to(K.dtype.element_ty), mask=mask)
    tl.store(k_ptr + half + offs, (k1 * s + k2 * c).to(K.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self._Wqkv = None
        self._cos = None
        self._sin = None
        self._cache_key = None

    def _prepare(self, x):
        Sx, E = x.shape
        key = (x.device, x.dtype, Sx, E)
        if self._cache_key != key or self._Wqkv is None:
            # Fused projection weight: one big GEMM instead of three
            self._Wqkv = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).contiguous()
            half = E // 2
            pos = torch.arange(Sx, device=x.device, dtype=torch.float32).unsqueeze(1)
            freq = torch.exp(
                torch.arange(0, half, device=x.device, dtype=torch.float32)
                * (-math.log(10000.0) / max(half, 1))
            )
            ang = pos * freq
            self._cos = torch.cos(ang).contiguous()
            self._sin = torch.sin(ang).contiguous()
            self._cache_key = key

    def forward(self, x):
        Sx, E = x.shape
        self._prepare(x)

        # Single fused GEMM for q, k, v
        qkv = x @ self._Wqkv  # (S, 3E)
        q = qkv[:, :E]
        k = qkv[:, E:2 * E]
        v = qkv[:, 2 * E:]

        half = E // 2
        if x.is_cuda and half > 0:
            BLOCK = max(triton.next_power_of_2(half), 16)
            _rope_qk_kernel[(Sx,)](
                q, k, self._cos, self._sin,
                half,
                qkv.stride(0), qkv.stride(0),
                BLOCK=BLOCK,
                num_warps=4,
            )
        else:
            pos_cos, pos_sin = self._cos, self._sin
            q1 = q[:, :half].float(); q2 = q[:, half:2 * half].float()
            k1 = k[:, :half].float(); k2 = k[:, half:2 * half].float()
            q[:, :half] = (q1 * pos_cos - q2 * pos_sin).to(q.dtype)
            q[:, half:2 * half] = (q1 * pos_sin + q2 * pos_cos).to(q.dtype)
            k[:, :half] = (k1 * pos_cos - k2 * pos_sin).to(k.dtype)
            k[:, half:2 * half] = (k1 * pos_sin + k2 * pos_cos).to(k.dtype)

        # Fused causal attention (flash attention, fp32 accumulation)
        out = F.scaled_dot_product_attention(
            q.unsqueeze(0).unsqueeze(0),
            k.unsqueeze(0).unsqueeze(0),
            v.unsqueeze(0).unsqueeze(0),
            is_causal=True,
        )
        return out[0, 0]

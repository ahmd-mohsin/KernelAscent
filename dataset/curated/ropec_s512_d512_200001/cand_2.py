import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 200001
S, D, DT = 512, 512, torch.float16


def _rope_ref(t):
    S_, E = t.shape
    half = E // 2
    pos = torch.arange(S_, device=t.device, dtype=torch.float32).unsqueeze(1)
    freq = torch.exp(torch.arange(0, half, device=t.device, dtype=torch.float32) * (-math.log(10000.0) / max(half, 1)))
    ang = pos * freq
    cos, sin = torch.cos(ang), torch.sin(ang)
    t1 = t[..., :half].float(); t2 = t[..., half:half * 2].float()
    out = t.float().clone()
    out[..., :half] = t1 * cos - t2 * sin
    out[..., half:half * 2] = t1 * sin + t2 * cos
    return out.to(t.dtype)


@triton.jit
def _rope_qk_kernel(
    QKV,            # (S, 3*Dm) fp16, rope applied in-place to first 2*Dm cols
    COS, SIN,       # (S, half) fp32
    half,           # Dm // 2
    Dm,             # model dim
    stride_row,     # row stride of QKV
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    which = tl.program_id(1)   # 0 -> q, 1 -> k
    offs = tl.arange(0, BLOCK)
    mask = offs < half

    base = QKV + row * stride_row + which * Dm
    t1 = tl.load(base + offs, mask=mask, other=0.0).to(tl.float32)
    t2 = tl.load(base + half + offs, mask=mask, other=0.0).to(tl.float32)

    c = tl.load(COS + row * half + offs, mask=mask, other=0.0)
    s = tl.load(SIN + row * half + offs, mask=mask, other=0.0)

    o1 = t1 * c - t2 * s
    o2 = t1 * s + t2 * c

    tl.store(base + offs, o1.to(QKV.dtype.element_ty), mask=mask)
    tl.store(base + half + offs, o2.to(QKV.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def _get_cache(self, seq_len, Dm, device):
        cache = getattr(self, "_cache", None)
        if (cache is None or cache[0] != seq_len or cache[1] != Dm
                or cache[2].device != device):
            half = Dm // 2
            pos = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
            freq = torch.exp(
                torch.arange(0, half, device=device, dtype=torch.float32)
                * (-math.log(10000.0) / max(half, 1))
            )
            ang = pos * freq
            cos = torch.cos(ang).contiguous()
            sin = torch.sin(ang).contiguous()
            Wcat = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).contiguous()
            self._cache = (seq_len, Dm, cos, sin, Wcat)
        return self._cache

    def forward(self, x):
        if not x.is_cuda:
            q = _rope_ref(x @ self.Wq); k = _rope_ref(x @ self.Wk); v = x @ self.Wv
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
            scores = scores + torch.triu(torch.full_like(scores, float('-inf')), diagonal=1)
            a = torch.softmax(scores, dim=-1)
            return a @ v

        seq_len, Dm = x.shape
        _, _, cos, sin, Wcat = self._get_cache(seq_len, Dm, x.device)

        # Fused single GEMM for Q, K, V
        qkv = x @ Wcat  # (S, 3*Dm), contiguous

        half = Dm // 2
        BLOCK = triton.next_power_of_2(half)
        _rope_qk_kernel[(seq_len, 2)](
            qkv, cos, sin, half, Dm, qkv.stride(0), BLOCK=BLOCK,
            num_warps=4,
        )

        q = qkv[:, :Dm]
        k = qkv[:, Dm:2 * Dm]
        v = qkv[:, 2 * Dm:]

        out = F.scaled_dot_product_attention(
            q.unsqueeze(0).unsqueeze(0),
            k.unsqueeze(0).unsqueeze(0),
            v.unsqueeze(0).unsqueeze(0),
            is_causal=True,
        )
        return out[0, 0]

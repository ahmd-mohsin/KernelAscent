import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 200020
S, D, DT = 1024, 2048, torch.float16


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
def _rope_kernel(X, COS, SIN, half, stride_x, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    pid = tl.program_id(1)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < half
    base = X + row * stride_x
    x1 = tl.load(base + offs, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(base + half + offs, mask=mask, other=0.0).to(tl.float32)
    c = tl.load(COS + row * half + offs, mask=mask, other=0.0)
    s = tl.load(SIN + row * half + offs, mask=mask, other=0.0)
    o1 = x1 * c - x2 * s
    o2 = x1 * s + x2 * c
    tl.store(base + offs, o1, mask=mask)
    tl.store(base + half + offs, o2, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def _get_cache(self, S_, E, device):
        key = (S_, E, device)
        if getattr(self, "_rope_key", None) != key:
            half = E // 2
            pos = torch.arange(S_, device=device, dtype=torch.float32).unsqueeze(1)
            freq = torch.exp(torch.arange(0, half, device=device, dtype=torch.float32)
                             * (-math.log(10000.0) / max(half, 1)))
            ang = pos * freq
            self._cos = torch.cos(ang).contiguous()
            self._sin = torch.sin(ang).contiguous()
            self._rope_key = key
        if getattr(self, "_wqkv_key", None) != (device, self.Wq.dtype):
            self._Wqkv = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).contiguous()
            self._wqkv_key = (device, self.Wq.dtype)
        return self._cos, self._sin, self._Wqkv

    def forward(self, x):
        if not x.is_cuda:
            q = _rope_ref(x @ self.Wq); k = _rope_ref(x @ self.Wk); v = x @ self.Wv
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
            a = torch.softmax(scores, dim=-1)
            return a @ v

        S_, E = x.shape
        cos, sin, Wqkv = self._get_cache(S_, E, x.device)

        qkv = x @ Wqkv  # (S, 3E), one fused GEMM
        q = qkv[:, :E]
        k = qkv[:, E:2 * E]
        v = qkv[:, 2 * E:]

        half = E // 2
        if half > 0:
            BLOCK = 256
            grid = (S_, triton.cdiv(half, BLOCK))
            _rope_kernel[grid](q, cos, sin, half, qkv.stride(0), BLOCK=BLOCK)
            _rope_kernel[grid](k, cos, sin, half, qkv.stride(0), BLOCK=BLOCK)

        out = F.scaled_dot_product_attention(
            q.unsqueeze(0).unsqueeze(0),
            k.unsqueeze(0).unsqueeze(0),
            v.unsqueeze(0).unsqueeze(0),
            scale=1.0 / math.sqrt(E),
        )
        return out.squeeze(0).squeeze(0)

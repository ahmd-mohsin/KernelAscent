import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False

SEED = 200014
S, D, DT = 1024, 512, torch.bfloat16


if _HAS_TRITON:
    @triton.jit
    def _rope_qk_kernel(
        qkv_ptr, cos_ptr, sin_ptr,
        stride_row,
        HALF: tl.constexpr, DIM: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, HALF)
        c = tl.load(cos_ptr + row * HALF + offs)
        s = tl.load(sin_ptr + row * HALF + offs)

        base = qkv_ptr + row * stride_row
        # q
        q1 = tl.load(base + offs).to(tl.float32)
        q2 = tl.load(base + HALF + offs).to(tl.float32)
        tl.store(base + offs, q1 * c - q2 * s)
        tl.store(base + HALF + offs, q1 * s + q2 * c)
        # k
        k1 = tl.load(base + DIM + offs).to(tl.float32)
        k2 = tl.load(base + DIM + HALF + offs).to(tl.float32)
        tl.store(base + DIM + offs, k1 * c - k2 * s)
        tl.store(base + DIM + HALF + offs, k1 * s + k2 * c)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self._Wcat = None
        self._trig_cache = {}

    def _get_wcat(self, device):
        if self._Wcat is None or self._Wcat.device != device:
            self._Wcat = torch.cat(
                [self.Wq.to(device), self.Wk.to(device), self.Wv.to(device)], dim=1
            ).contiguous()
        return self._Wcat

    def _get_trig(self, seq_len, device):
        key = (seq_len, device)
        if key not in self._trig_cache:
            half = D // 2
            pos = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
            freq = torch.exp(
                torch.arange(0, half, device=device, dtype=torch.float32)
                * (-math.log(10000.0) / max(half, 1))
            )
            ang = pos * freq
            self._trig_cache[key] = (torch.cos(ang).contiguous(), torch.sin(ang).contiguous())
        return self._trig_cache[key]

    def forward(self, x):
        seq_len, dim = x.shape
        device = x.device
        Wcat = self._get_wcat(device)
        cos, sin = self._get_trig(seq_len, device)

        # Fused QKV projection: one matmul instead of three
        qkv = x @ Wcat  # (S, 3D), contiguous

        half = dim // 2
        if _HAS_TRITON and x.is_cuda:
            _rope_qk_kernel[(seq_len,)](
                qkv, cos, sin,
                qkv.stride(0),
                HALF=half, DIM=dim,
                num_warps=4,
            )
            q = qkv[:, :dim]
            k = qkv[:, dim:2 * dim]
            v = qkv[:, 2 * dim:]
        else:
            qf = qkv[:, :dim].float()
            kf = qkv[:, dim:2 * dim].float()
            q1, q2 = qf[:, :half], qf[:, half:]
            k1, k2 = kf[:, :half], kf[:, half:]
            q = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1).to(x.dtype)
            k = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1).to(x.dtype)
            v = qkv[:, 2 * dim:]

        scores = (q @ k.transpose(-1, -2)) / math.sqrt(dim)
        a = torch.softmax(scores, dim=-1)
        return a @ v

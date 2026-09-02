import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 554
M, D, DT = 8192, 2049, torch.bfloat16


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 1024, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)

    def forward(self, x):
        # softmax output is always >= 0, so both relu calls are identity ops -> drop them.
        #
        # K = 2049 is odd, which forces cuBLAS onto misaligned (slow) tensor-core paths.
        # Zero-pad the reduction dimension to a multiple of 8 (exact: adding 0*w terms
        # contributes exactly 0 to the fp32 accumulators), enabling fast aligned kernels.
        W = self.W0
        Wp = getattr(self, "_Wp", None)
        if Wp is None or Wp.device != x.device or Wp.dtype != W.dtype:
            K, N = W.shape
            Kp = ((K + 7) // 8) * 8
            Wp = torch.zeros(Kp, N, dtype=W.dtype, device=x.device)
            Wp[:K].copy_(W.to(x.device))
            object.__setattr__(self, "_Wp", Wp)

        Kp = Wp.shape[0]
        K = x.shape[-1]
        if Kp != K:
            xp = F.pad(x, (0, Kp - K))
        else:
            xp = x

        y = xp @ Wp
        return torch.softmax(y, dim=-1)

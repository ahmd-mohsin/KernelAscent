# 0-score failure forensics — where every model family gets stuck (mechanistic)

For each model we sample K candidates on every task and classify **why** each scored 0. The 0-score 'correctness wall' below ~3B is really a **kernel-formation wall**: models fail before producing a valid kernel (`no_extract`, `syntax_error`), not by writing runnable-but-wrong kernels. As scale rises the dominant failure moves *downstream*: incoherence → truncation → API-hallucination → wrong-output → correct. Each scale step advances the model one stage down the pipeline.

| model | size | zero% | no_extract | syntax | name/API | wrong_out | correct | dominant stage |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-Coder-0.5B | 0.5B | 100% | 71% | 29% | 0% | 0% | 0% | incoherence/format |
| DeepSeek-Coder-1.3B | 1.3B | 86% | 90% | 7% | 1% | 0% | 1% | incoherence/format |
| Qwen2.5-Coder-1.5B | 1.5B | 90% | 57% | 39% | 1% | 1% | 1% | incoherence/format |
| Yi-Coder-1.5B | 1.5B | 97% | 96% | 3% | 0% | 0% | 0% | incoherence/format |
| Qwen2.5-Coder-3B | 3B | 48% | 51% | 33% | 5% | 2% | 6% | incoherence/format |
| DeepSeek-Coder-6.7B | 6.7B | 62% | 80% | 11% | 3% | 0% | 3% | incoherence/format |
| Qwen2.5-Coder-7B | 7B | 28% | 8% | 75% | 5% | 0% | 9% | truncation/syntax |
| Qwen2.5-Coder-14B | 14B | 14% | 0% | 9% | 8% | 8% | 65% | correct |

## Qwen2.5-Coder-0.5B (0.5B) — example failure chains
**no_extract** (incoherence/format) — task `l3_2`, error: `-`
```
...36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36"
 userAgent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36"
 userAgent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88
```
**syntax_error** (truncation/syntax) — task `l3_2`, error: `SyntaxError('invalid syntax', ('/home/greenland-user/ka/ka_data/cand_modules/can`
```
...     H, HKV, HD, D = self.N_HEADS, self.N_KV_HEADS, self.HEAD_DIM, self.D

        h = self._rmsnorm(x, self.norm1_w)
        qkv = h @ self.w_qkv.t()                                     # [B, S, (H+2*HKV)*HD]
        qkv = torch.cat([h, qkv], dim=-1)                           # [B, S, (2H+2*HKV)*HD
```
**name_error** (API-hallucination) — task `l2_14`, error: `NameError("name 'get_default_device' is not defined")`
```
....device)
        self.input.gather(1, self.outputs)
        return self.outputs

def get_inputs():
    x = torch.randn(8, 2048, 4096)
    return [x.to(get_default_device())]

def get_outputs():
    x = torch.randn(8, 2048, 4096)
    return [x.to(get_default_device())]

def get_compiler():
    return
```

## DeepSeek-Coder-1.3B (1.3B) — example failure chains
**no_extract** (incoherence/format) — task `l3_2`, error: `-`
```
...f.sin[:s][None, None]
    qf, kf = q.float(), k.float()
    q = (qf * cos + _rotate_half(qf) * sin).to(x.dtype)
    k = (kf * cos + _rotate_half(kf) * sin).to(x.dtype)

    rep = h // hkv
    k = k.repeat_interleave(rep, dim=1)
    v = v.repeat_interleave(rep, dim=1)

    scores = torch.matmul(q, k.
```
**syntax_error** (truncation/syntax) — task `l3_2`, error: `SyntaxError("invalid character '💬' (U+1F4AC)", ('/home/greenland-user/ka/ka_data`
```
...lNew(nn.Module)` with the same __init__/forward signature, in ONE python code block.
  💬
"""
    )
    print(
        """
💬
  It is your job to rewrite the code of the previous module in ONE python block,
  as described in the task description.
"""
    )
    print(
        """
💬
  Your solution must
```
**name_error** (API-hallucination) — task `l2_14`, error: `AttributeError("'NoneType' object has no attribute 'shape'")`
```
...er(w.to(device=dev, dtype=dt), requires_grad=False)
        self.bias = nn.Parameter(b.to(device=dev, dtype=dt), requires_grad=False)
        self.scale = nn.Parameter(s.to(device=dev, dtype=dt), requires_grad=False)

    def forward(self, x):
        h = F.linear(x, self.weight, self.bias)      # G
```
**wrong_output** (silently-wrong) — task `l2_15`, error: `wrong/imprecise`
```
...

def f3(x):
    x = F.relu(x)
    x = F.sigmoid(x)
    return x


# gradient checks


def g(x):
    return torch.randn_like(x)


def g2(x):
    x = F.relu(x)
    return torch.randn_like(x)


def g3(x):
    x = F.relu(x)
    x = F.sigmoid(x)
    return torch.randn_like(x)


def test_f():
    x = tor
```

## Qwen2.5-Coder-1.5B (1.5B) — example failure chains
**no_extract** (incoherence/format) — task `l3_2`, error: `-`
```
...Implement the exact same functionality using triton. Avoid using triton's `@triton.jit` decorator; instead use the triton autotuner to get a JIT-optimized kernel.
```
**syntax_error** (truncation/syntax) — task `l3_2`, error: `SyntaxError('invalid syntax', ('/home/greenland-user/ka/ka_data/cand_modules/can`
```
...D_DIM, self.D

        h = self._rmsnorm(x, self.norm1_w)
        qkv = h @ self.w_qkv.t()                                     # [B, S, (H+2*HKV)*HD]
        q, k, v = torch.split(qkv, [H * HD, HKV * HD, HKV * HD], dim=-1)
        q = q.view(b, s, H, HD).transpose(1, 2)                      # [B, H,
```
**name_error** (API-hallucination) — task `l1_1`, error: `ImportError("cannot import name 'cuda' from 'triton' (/usr/local/lib/python3.12/`
```
...       with cuda.device(0):
            dt = x.dtype
            y = x.float()
            ms = y.square().mean(dim=-1, keepdim=True)
            y = y * tl.rsqrt(ms + self.eps)
            y = y * self.weight + self.bias
            out = y * tl.sigmoid(y * self.gate)
            return out.to(dt)

```
**wrong_output** (silently-wrong) — task `l2_14`, error: `wrong/imprecise`
```
...er(torch.randn(1).to(device="cuda", dtype=dt), requires_grad=False)

    def forward(self, x):
        h = F.linear(x, self.weight, self.bias)  # GEMM + bias
        h = F.gelu(h, approximate="tanh")        # activation
        return x + h * self.scale                # per-channel scale + residual

```

## Yi-Coder-1.5B (1.5B) — example failure chains
**no_extract** (incoherence/format) — task `l3_2`, error: `-`
```
....append(torch.randn(b, m, k, dtype=dtype))
    return inputs

def bench_cpu(f, inputs, dtype):
    with benchmark() as b:
        for x in inputs:
            f(x)
    return b.average_time() * (1e-6 * x.shape[0])

def bench_gpu(f, inputs, dtype):
    with benchmark() as b:
        for x in inputs:

```
**syntax_error** (truncation/syntax) — task `l2_14`, error: `SyntaxError("'(' was never closed", ('/home/greenland-user/ka/ka_data/cand_modul`
```
...s_strided(self.model.bias.size(), (1, 1, 1, 1, 1, 1)),
            self.model.scale.contiguous().as_strided(self.model.scale.size(), (1, 1, 1, 1, 1, 1)),
            x.contiguous().as_strided(x.size(), (1, 1, 1, 1, K, 1)),
            x_stride=1, W_stride=4, b_stride=8, scale_stride=12, y_stride=16,
```
**name_error** (API-hallucination) — task `l2_1`, error: `NameError("name 'g' is not defined")`
```
... return elapsed * 1e3

if __name__ == "__main__":
    print("FP16")
    model = Model().float()
    print(time_module(model))
    print(time_fuse(model.forward))
    print(time_fuse_torch(model.forward))
    print(time_fuse_triton(model.forward))
    print(time_fuse_triton_torch(model.forward))
    
```

## Qwen2.5-Coder-3B (3B) — example failure chains
**no_extract** (incoherence/format) — task `l3_2`, error: `-`
```
...
        k = k.view(b, s, HKV, HD).transpose(1, 2)                    # [B, HKV, S, HD]
        v = v.view(b, s, HKV, HD).transpose(1, 2)

        cos = self.cos[:s][None, None]
        sin = self.sin[:s][None, None]
        qf, kf = q.float(), k.float()
        q = (qf * cos + self._rotate_half(qf)
```
**syntax_error** (truncation/syntax) — task `l3_2`, error: `SyntaxError('invalid syntax', ('/home/greenland-user/ka/ka_data/cand_modules/can`
```
....float()).to(t.dtype)

    def forward(self, x):
        b, s, _ = x.shape
        H, HKV, HD, D = self.N_HEADS, self.N_KV_HEADS, self.HEAD_DIM, self.D

        h = self._rmsnorm(x, self.norm1_w)
        qkv = h @ self.w_qkv.t()                                     # [B, S, (H+2*HKV)*HD]
        q, k
```
**name_error** (API-hallucination) — task `l2_7`, error: `AttributeError("module 'torch' has no attribute 'gelu'")`
```
....bias
        # Use torch.gelu instead of F.gelu for performance
        h = torch.gelu(h, approximate="tanh")
        return x + self.gamma * h


def get_inputs():
    g = torch.Generator().manual_seed(42)
    x = torch.randn(BATCH, SEQ, DIM, generator=g)
    return [x.to(device=_DEVICE, dtype=DT)]
```
**wrong_output** (silently-wrong) — task `l2_11`, error: `wrong/imprecise`
```
...












































































































































































































































































































```

## DeepSeek-Coder-6.7B (6.7B) — example failure chains
**no_extract** (incoherence/format) — task `l3_2`, error: `-`
```
...es = scores - 1e10 * mask
    p = tl.softmax(scores, axis=-1)
    o = p @ v
    out[..., :, :] = o


@triton.jit
def fused_qkv_rope(q, k, v, mask, pb, out):
    # mask is [S, S]
    # pb is a bool
    # q, k, v are [B, S, D]
    # out is [B, N_HEADS, S, HEAD_DIM]
    # qkv is [B, S, N_HEADS*HEAD_DIM
```
**syntax_error** (truncation/syntax) — task `l2_7`, error: `SyntaxError("'(' was never closed", ('/home/greenland-user/ka/ka_data/cand_modul`
```
...nt(hf_model_out.dtype, hf_model_new_out.dtype)
    print(hf_model_out.eq(hf_model_new_out).all())
    print(torch.allclose(hf_model_out, hf_model_new_out, atol=0.01, rtol=0.01))

    input_ids_t = input_ids.to(model_new.hf_model.device)
    model_new.hf_model.to(model_new.hf_model.device)
    print(
```
**name_error** (API-hallucination) — task `l2_2`, error: `AttributeError("'NoneType' object has no attribute 'shape'")`
```
...


if __name__ == "__main__":
    model = Model()
    model.to(_DEVICE)
    model.eval()

    with torch.no_grad():
        x = get_inputs()
        y = model(x)
        z = model.forward(*x)
        print(y.device, y.dtype)
        print(z.device, z.dtype)
        print(torch.max(torch.abs(y - z)))
```
**wrong_output** (silently-wrong) — task `l2_14`, error: `wrong/imprecise`
```
...












































































































































































































































































































```

## Qwen2.5-Coder-7B (7B) — example failure chains
**no_extract** (incoherence/format) — task `l3_2`, error: `-`
```
...r("sin", emb.sin().to(dev), persistent=False)
        self.register_buffer(
            "mask", torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1).to(dev), persistent=False
        )

    @staticmethod
    def _rotate_half(t):
        t1, t2 = t.chunk(2, dim=-1)
        return torch.cat([-t2
```
**syntax_error** (truncation/syntax) — task `l3_2`, error: `SyntaxError('invalid syntax', ('/home/greenland-user/ka/ka_data/cand_modules/can`
```
...D_DIM, self.D

        h = self._rmsnorm(x, self.norm1_w)
        qkv = h @ self.w_qkv.t()                                     # [B, S, (H+2*HKV)*HD]
        q, k, v = torch.split(qkv, [H * HD, HKV * HD, HKV * HD], dim=-1)
        q = q.view(b, s, H, HD).transpose(1, 2)                      # [B, H,
```
**name_error** (API-hallucination) — task `l3_2`, error: `NameError("name 'triton' is not defined")`
```
...t * s * s, mask=tl.arange(0, s) < s)
        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin
        q = q.reshape(num_warps, block_size // num_warps, num_heads, head_dim)
        k = k.reshape(num_warps, block_size // num_warps, num_heads, head_dim)
       
```
**wrong_output** (silently-wrong) — task `l1_6`, error: `wrong/imprecise`
```
....eps) * self.w_in.float() + self.b_in.float()
        a = 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
        y = x + a * self.w_res.float()
        y = y * torch.rsqrt(y.pow(2).sum(dim=-1, keepdim=True) + self.eps) * self.w_out.float()
        return y.to(x.dtype)

```

## Qwen2.5-Coder-14B (14B) — example failure chains
**syntax_error** (truncation/syntax) — task `l3_2`, error: `SyntaxError("'(' was never closed", ('/home/greenland-user/ka/ka_data/cand_modul`
```
...ose(1, 2)                    # [B, HKV, S, HD]
        v = v.view(b, s, HKV, HD).transpose(1, 2)

        cos = self.cos[:s][None, None]
        sin = self.sin[:s][None, None]
        qf, kf = q.float(), k.float()
        q = (qf * cos + self._rotate_half(qf) * sin).to(x.dtype)
        k = (kf * cos
```
**name_error** (API-hallucination) — task `l2_2`, error: `ImportError("cannot import name '_amp_foreach_non_fused_bias_gelu' from 'torch._`
```
...non_fused_bias_gelu([x], [self.weight], [self.bias])[0]
        y = y * self.gamma  # per-channel scale
        return x + y         # residual


def get_inputs():
    g = torch.Generator(device="cpu").manual_seed(1)
    x = torch.randn(M, K, generator=g).to(device=_DEVICE, dtype=DT)
    return [x]

```
**wrong_output** (silently-wrong) — task `l2_7`, error: `wrong/imprecise`
```
...orward(self, x):
        h = F.linear(x, self.weight.t(), self.bias)
        h = F.gelu(h, approximate="tanh")
        return x + self.gamma * h

def get_inputs():
    g = torch.Generator().manual_seed(42)
    x = torch.randn(BATCH, SEQ, DIM, generator=g)
    return [x.to(device=_DEVICE, dtype=DT)]

```

#!/usr/bin/env python3
"""P3 — MUTATION-DSL TASK-BANK GENERATOR. Parametrically synthesizes hundreds of self-contained, gradeable
GPU-kernel tasks by crossing OP-FAMILY x SHAPE x DTYPE (+ per-family variant). Each emitted `source` matches the
grader contract (agent_bench.build_ref): defines DT, class Model(__init__(self, dt)) on cuda, get_inputs(), forward.
This (a) fixes the too-small-bank probe-AUC instability (29 -> ~400 tasks, split by FAMILY for leakage-resistant
CV) and (b) supplies the self-play author's structured mutation space (vs from-scratch prose). Writes a JSON bank
[{name, source, tier, family}]; point KA_KERNEL_BANK at it. Run the learnability gate next to keep solvable tasks.

Usage: python3 scripts/gen_task_dsl.py --out dataset/kernel_bank/kernel_tasks_dsl.json
"""
import os, json, argparse, itertools

HEAD = "import torch\nimport torch.nn as nn\nDT = torch.{dt}\nROWS, HIDDEN = {rows}, {hidden}\n\n"

# each family: name -> (init_body, forward_body). Params built with dtype `dt`; input [ROWS,HIDDEN] of DT.
FAMILIES = {
 "rmsnorm_gate": ("""self.eps={eps}
        self.w=nn.Parameter((1.0+0.1*torch.randn(HIDDEN,generator=g)).to(dev,dt))""",
   """s=x.float(); v=s.pow(2).mean(-1,keepdim=True); s=s*torch.rsqrt(v+self.eps)
        s=s.to(x.dtype)*self.w; return s*torch.sigmoid(s)"""),
 "layernorm_affine": ("""self.eps={eps}
        self.w=nn.Parameter((1.0+0.1*torch.randn(HIDDEN,generator=g)).to(dev,dt))
        self.b=nn.Parameter((0.1*torch.randn(HIDDEN,generator=g)).to(dev,dt))""",
   """s=x.float(); m=s.mean(-1,keepdim=True); v=s.var(-1,keepdim=True,unbiased=False)
        s=(s-m)*torch.rsqrt(v+self.eps); return (s.to(x.dtype)*self.w+self.b)"""),
 "softmax_row": ("""self.sc={eps}""",
   """s=(x.float()*self.sc); s=s-s.amax(-1,keepdim=True); e=s.exp(); return (e/e.sum(-1,keepdim=True)).to(x.dtype)"""),
 "gelu_mlp": ("""self.w1=nn.Parameter((0.02*torch.randn(HIDDEN,HIDDEN,generator=g)).to(dev,dt))""",
   """h=x@self.w1; return (0.5*h*(1.0+torch.tanh(0.7978845608*(h+{eps}*h*h*h))))"""),
 "matmul_bias": ("""self.w=nn.Parameter((0.02*torch.randn(HIDDEN,HIDDEN,generator=g)).to(dev,dt))
        self.b=nn.Parameter((0.1*torch.randn(HIDDEN,generator=g)).to(dev,dt))""",
   """return torch.relu(x@self.w+self.b*{eps})"""),
 "fused_elemwise": ("""self.a=nn.Parameter((0.1*torch.randn(HIDDEN,generator=g)).to(dev,dt))
        self.b=nn.Parameter((0.1*torch.randn(HIDDEN,generator=g)).to(dev,dt))""",
   """return torch.relu(x*self.a+self.b)*torch.sigmoid(x*{eps})"""),
 "sigmoid_gate": ("""self.w=nn.Parameter((1.0+0.1*torch.randn(HIDDEN,generator=g)).to(dev,dt))""",
   """return (x*torch.sigmoid(x*{eps}))*self.w"""),
 "softplus_norm": ("""self.eps={eps}""",
   """s=torch.nn.functional.softplus(x.float()); return (s/(s.mean(-1,keepdim=True)+self.eps)).to(x.dtype)"""),
 # --- families added for FAMILY-DISJOINT splits (P2). Holding out whole families is only a real
 # transfer test if the held-out computation is structurally different, not a reshaped twin, so each
 # of these introduces a distinct reduction/access pattern rather than another elementwise chain.
 "swiglu_mlp": ("""self.wg=nn.Parameter((0.02*torch.randn(HIDDEN,HIDDEN,generator=g)).to(dev,dt))
        self.wu=nn.Parameter((0.02*torch.randn(HIDDEN,HIDDEN,generator=g)).to(dev,dt))""",
   """a=x@self.wg; b=x@self.wu; return (a*torch.sigmoid(a*{eps}))*b"""),
 "groupnorm_rows": ("""self.eps={eps}
        self.w=nn.Parameter((1.0+0.1*torch.randn(HIDDEN,generator=g)).to(dev,dt))""",
   """s=x.float().view(x.shape[0], 8, -1); m=s.mean(-1,keepdim=True); v=s.var(-1,keepdim=True,unbiased=False)
        s=((s-m)*torch.rsqrt(v+self.eps)).view_as(x); return s.to(x.dtype)*self.w"""),
 "logsumexp_norm": ("""self.sc={eps}""",
   """s=x.float()*self.sc; return (s-torch.logsumexp(s,dim=-1,keepdim=True)).to(x.dtype)"""),
 "cumsum_scale": ("""self.sc={eps}""",
   """s=x.float().cumsum(-1)*self.sc; return (s/(1.0+s.abs())).to(x.dtype)"""),
 "topk_mask": ("""self.k=int({eps})""",
   """v,_=x.float().topk(self.k,dim=-1); thr=v[...,-1:].contiguous()
        return torch.where(x.float()>=thr, x.float(), torch.zeros((),device=x.device)).to(x.dtype)"""),
 "softmax_matmul": ("""self.w=nn.Parameter((0.02*torch.randn(HIDDEN,HIDDEN,generator=g)).to(dev,dt))
        self.sc={eps}""",
   """s=(x.float()*self.sc); s=s-s.amax(-1,keepdim=True); p=(s.exp()); p=p/p.sum(-1,keepdim=True)
        return (p.to(x.dtype))@self.w"""),
 "rope_pair": ("""self.sc={eps}""",
   """h=x.shape[-1]//2; a,b=x.float()[...,:h],x.float()[...,h:]
        i=torch.arange(h,device=x.device,dtype=torch.float32)*self.sc
        c,sn=torch.cos(i),torch.sin(i); return torch.cat([a*c-b*sn, a*sn+b*c],dim=-1).to(x.dtype)"""),
 "var_gate": ("""self.eps={eps}
        self.w=nn.Parameter((1.0+0.1*torch.randn(HIDDEN,generator=g)).to(dev,dt))""",
   """s=x.float(); v=s.var(-1,keepdim=True,unbiased=False); g_=torch.rsqrt(v+self.eps)
        return (s*g_).to(x.dtype)*self.w*torch.tanh(x)"""),
}

TMPL = HEAD + """class Model(nn.Module):
    def __init__(self, dt=DT, hidden=HIDDEN):
        super().__init__()
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        g = torch.Generator().manual_seed(0)
        {init}
    def forward(self, x):
        {fwd}

def get_inputs():
    g = torch.Generator().manual_seed(1)
    return [(torch.randn(ROWS, HIDDEN, generator=g)).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"), DT)]
"""

SHAPES = [(8192, 2048), (16384, 4096), (32768, 4096), (4096, 8192), (16384, 2048)]
DTYPES = ["float16", "bfloat16", "float32"]
EPS = {"rmsnorm_gate": [1e-5, 1e-6], "layernorm_affine": [1e-5, 1e-6], "softmax_row": [1.0, 0.5],
       "gelu_mlp": [0.044715, 0.05], "matmul_bias": [1.0, 0.5], "fused_elemwise": [1.0, 0.7],
       "sigmoid_gate": [1.0, 1.702], "softplus_norm": [1e-5, 1e-6],
       "swiglu_mlp": [1.0, 1.702], "groupnorm_rows": [1e-5, 1e-6], "logsumexp_norm": [1.0, 0.5],
       "cumsum_scale": [1.0, 0.25], "topk_mask": [8, 32], "softmax_matmul": [1.0, 0.5],
       "rope_pair": [1e-4, 5e-4], "var_gate": [1e-5, 1e-6]}

# O(n*h^2) families need the hidden dim capped so a single grade stays affordable.
QUADRATIC = ("gelu_mlp", "matmul_bias", "swiglu_mlp", "softmax_matmul")
TIER2 = QUADRATIC

# FAMILY-DISJOINT SPLIT (P2). Fixed here, never drawn at random, so the split is auditable and stable
# across regenerations. `train` families are what a probe/learner may fit on; `heldin` are unseen
# shape/dtype cells of TRAIN families (in-family generalisation); `heldout` families never appear in
# training at all (out-of-family transfer -- the real moat). A renamed template or a nearby shape is
# NOT a new family, so held-out families are chosen for structurally distinct access patterns.
FAMILY_SPLIT = {
    "train":   ["rmsnorm_gate", "layernorm_affine", "softmax_row", "gelu_mlp",
                "matmul_bias", "fused_elemwise", "sigmoid_gate", "softplus_norm"],
    "heldout": ["swiglu_mlp", "groupnorm_rows", "logsumexp_norm", "cumsum_scale",
                "topk_mask", "softmax_matmul", "rope_pair", "var_gate"],
}
# within TRAIN families, these dtype/shape cells are reserved as the in-family generalisation split
HELDIN_DTYPE = "float32"
HELDIN_SHAPE = (16384, 2048)


def split_of(fam, dt, shape):
    """Deterministic split label. Out-of-family transfer is the strong test; in-family generalisation
    (unseen dtype/shape of a seen family) is the weak one; everything else is trainable."""
    if fam in FAMILY_SPLIT["heldout"]:
        return "heldout_family"
    if dt == HELDIN_DTYPE or tuple(shape) == HELDIN_SHAPE:
        return "heldin_cell"
    return "train"


def gen(args):
    bank = []
    for fam, (initb, fwdb) in FAMILIES.items():
        for dt, (rows, hidden), eps in itertools.product(DTYPES, SHAPES, EPS[fam]):
            # O(n*h^2) families: cap hidden to avoid OOM/timeout
            if fam in QUADRATIC and hidden > 4096: continue
            src = TMPL.format(dt=dt, rows=rows, hidden=hidden,
                              init=initb.format(eps=eps), fwd=fwdb.format(eps=eps))
            name = "dsl_%s_%s_%dx%d_%s" % (fam, dt.replace("float", "f"), rows, hidden, str(eps).replace(".", "p").replace("-", "m"))
            tier = "L2" if fam in TIER2 else "L1"
            bank.append({"name": name, "source": src, "tier": tier, "family": fam,
                         "dtype": dt, "shape": [rows, hidden], "eps": eps,
                         "split": split_of(fam, dt, (rows, hidden))})
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(bank, open(args.out, "w"), indent=2)
    fams, splits = {}, {}
    for b in bank:
        fams[b["family"]] = fams.get(b["family"], 0) + 1
        splits[b["split"]] = splits.get(b["split"], 0) + 1
    print("wrote %d tasks over %d families -> %s" % (len(bank), len(fams), args.out))
    print("by split:", splits)
    print("by family:", fams)
    print("\nNOTE: these sources are generated, not yet validated. Run")
    print("  python3 scripts/validate_bank_cpu.py --in %s --out <validated>.json" % args.out)
    print("to apply the CPU semantic/degeneracy gate, then the GPU learnability gate.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset/kernel_bank/kernel_tasks_dsl.json")
    gen(ap.parse_args())

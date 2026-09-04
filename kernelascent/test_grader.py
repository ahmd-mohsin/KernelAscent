"""Harness correctness self-test (run on a GPU box). Plants known candidates for one task
and asserts the grader classifies each correctly and never dies on a crashing kernel.
Exit 0 = all invariants hold. This is the anti-bug gate for the scoring path.
"""
import os, sys, json, subprocess, shutil, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
GRADER = os.path.join(HERE, "grade_candidates.py")

TASK = '''import torch, torch.nn as nn
SEED = 123
M, D, DT = 512, 512, torch.float16
class Model(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
    def forward(self, x):
        return torch.softmax(x, dim=-1)
def get_inputs():
    g = torch.Generator().manual_seed(SEED + 1)
    return [torch.randn(M, D, generator=g).to(DT)]
'''

CANDS = {
    # correct: numerically equivalent, input-sensitive -> must PASS
    "cand_correct.py": "import torch, torch.nn as nn\nclass ModelNew(nn.Module):\n    def __init__(self, dtype=DT):\n        super().__init__()\n    def forward(self, x):\n        return torch.softmax(x, dim=-1)\n",
    # wrong output: returns input unchanged -> must FAIL (wrong_on)
    "cand_wrong.py": "import torch, torch.nn as nn\nclass ModelNew(nn.Module):\n    def __init__(self, dtype=DT):\n        super().__init__()\n    def forward(self, x):\n        return x\n",
    # reward hack: constant output ignores input -> must FAIL (input_insensitive)
    "cand_constant.py": "import torch, torch.nn as nn\nclass ModelNew(nn.Module):\n    def __init__(self, dtype=DT):\n        super().__init__()\n    def forward(self, x):\n        return torch.zeros_like(x)\n",
    # python error at construct/forward -> must FAIL, must NOT kill the run
    "cand_error.py": "import torch, torch.nn as nn\nclass ModelNew(nn.Module):\n    def __init__(self, dtype=DT):\n        super().__init__()\n    def forward(self, x):\n        return x @ x @ x @ x.reshape(1,2,3)\n",
}


def main():
    root = tempfile.mkdtemp(prefix="ka_gtest_")
    d = os.path.join(root, "softmax_test"); os.makedirs(d)
    open(d + "/task.py", "w").write(TASK)
    json.dump({"name": "softmax_test", "tier": "Easy", "family": "norm-act"}, open(d + "/meta.json", "w"))
    for fn, code in CANDS.items():
        open(os.path.join(d, fn), "w").write(code)
    # grade this one task dir (crash-isolated --one path)
    subprocess.run([sys.executable, "-u", GRADER, "--candir", root, "--one", d, "--cand-timeout", "60"], timeout=300)
    r = json.load(open(d + "/results.json"))
    by = {c["file"]: c for c in r.get("candidates", [])}
    print("candidate outcomes:")
    for f, c in by.items():
        print("  %-16s ok=%s reason=%s" % (f, c["ok"], c["reason"]))

    ok = True
    def check(cond, msg):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + msg); ok = ok and cond

    check(by.get("cand_correct.py", {}).get("ok") is True, "correct candidate passes")
    check(by.get("cand_wrong.py", {}).get("ok") is False, "wrong-output candidate fails")
    check(by.get("cand_constant.py", {}).get("ok") is False, "constant/reward-hack candidate rejected")
    check("insensitive" in str(by.get("cand_constant.py", {}).get("reason", "")), "constant rejected via input-sensitivity")
    check(by.get("cand_error.py", {}).get("ok") is False, "erroring candidate fails without killing the run")
    check(r.get("pass_at_k") == 1, "exactly one candidate passes (pass_at_k==1)")
    check(len(by) == len(CANDS), "all candidates graded (run survived the erroring one)")
    shutil.rmtree(root, ignore_errors=True)
    print("\nHARNESS TEST: " + ("ALL PASS" if ok else "FAILURES PRESENT"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

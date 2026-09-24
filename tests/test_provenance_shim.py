#!/usr/bin/env python3
"""Run PROV.dump exactly as the real call sites call it.

This shim broke the experiment batch twice -- once by not accepting `indent=2`, once by
assuming its second argument was a path when the call sites pass `open(path,"w")`. Both times
the test I wrote exercised a call *I* invented, so it passed while every real site failed at
the final write, after the GPU work was done.

So this test does not invent calls. It extracts every `PROV.dump(...)` invocation from the
repository, rebuilds its argument shape, and runs it.
"""
import ast, glob, json, os, re, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from kernelascent import provenance as P


def call_shapes():
    """(file, lineno, second-arg source) for every PROV.dump* call in the repo.

    Covers dump AND dump_atomic. Matching only `dump` would have left every atomic call site
    unexercised at the moment they were introduced -- and converting those call sites is exactly
    what broke this shim the first two times (once on a missing `indent`, once on a path/handle
    mix-up). The third conversion also passed a PATH to a bare `json.dump`, which needs a file
    object; that one was caught by reading the diff, not by a test, which is the gap this closes.
    """
    out = []
    for f in glob.glob(os.path.join(ROOT, "**", "*.py"), recursive=True):
        if os.path.basename(f) in ("provenance.py", os.path.basename(__file__)):
            continue
        try:
            tree = ast.parse(open(f).read(), filename=f)
        except Exception:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if not (isinstance(fn, ast.Attribute) and fn.attr in ("dump", "dump_atomic")
                    and isinstance(fn.value, ast.Name) and fn.value.id == "PROV"):
                continue
            if len(node.args) < 2:
                continue
            second = ast.dump(node.args[1])
            kinds = sorted(k.arg for k in node.keywords if k.arg)
            shape = "handle" if "open" in second else "path"
            if fn.attr == "dump_atomic" and shape == "handle":
                raise AssertionError(
                    "%s:%d passes an open handle to dump_atomic, which needs a PATH to rename "
                    "onto -- and open(path,'w') has already truncated the file by then"
                    % (os.path.relpath(f, ROOT), node.lineno))
            out.append((os.path.relpath(f, ROOT), node.lineno, shape, kinds, fn.attr))
    return out


def main():
    shapes = call_shapes()
    assert shapes, "found no PROV.dump call sites -- the extractor is broken, not the code"
    d = tempfile.mkdtemp()
    print("exercising %d real call sites:" % len(shapes))
    seen = set()
    for relpath, lineno, kind, kinds, attr in shapes:
        f = os.path.join(d, "%s_%d.json" % (os.path.basename(relpath).replace(".", "_"), lineno))
        kw = {k: 2 for k in kinds if k == "indent"}
        fn = getattr(P, attr)
        for payload in ({"a": 1}, [1, 2, 3]):          # both payload types in use
            if kind == "handle":
                with open(f, "w") as fh:
                    fn(payload, fh, **kw)
            else:
                fn(payload, f, **kw)
            got = json.load(open(f))
            assert isinstance(got, type(payload)), (relpath, lineno, type(got))
            if isinstance(got, dict):
                assert "_provenance" in got, "%s:%d lost its provenance" % (relpath, lineno)
            if attr == "dump_atomic":
                assert not os.path.exists(f + ".tmp"), \
                    "%s:%d left a .tmp behind -- a crash here strands it forever" % (relpath, lineno)
        seen.add((attr, kind, tuple(kinds)))
        print("  ok  %-34s:%-4d  %-11s dest=%-6s kwargs=%s"
              % (relpath, lineno, attr, kind, kinds or "-"))
    print("\ndistinct call shapes covered: %s" % sorted(seen))
    assert any(a == "dump_atomic" for a, _, _ in seen), \
        "no dump_atomic call site exercised -- durable artifacts are not being written atomically"
    print("PASS -- every real PROV.dump/dump_atomic call site works with both payload types")
    return 0


if __name__ == "__main__":
    sys.exit(main())

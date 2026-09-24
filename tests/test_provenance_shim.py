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
    """(file, lineno, second-arg source) for every PROV.dump call in the repo."""
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
            if not (isinstance(fn, ast.Attribute) and fn.attr == "dump"
                    and isinstance(fn.value, ast.Name) and fn.value.id == "PROV"):
                continue
            if len(node.args) < 2:
                continue
            second = ast.dump(node.args[1])
            kinds = sorted(k.arg for k in node.keywords if k.arg)
            out.append((os.path.relpath(f, ROOT), node.lineno,
                        "handle" if "open" in second else "path", kinds))
    return out


def main():
    shapes = call_shapes()
    assert shapes, "found no PROV.dump call sites -- the extractor is broken, not the code"
    d = tempfile.mkdtemp()
    print("exercising %d real call sites:" % len(shapes))
    seen = set()
    for relpath, lineno, kind, kinds in shapes:
        f = os.path.join(d, "%s_%d.json" % (os.path.basename(relpath).replace(".", "_"), lineno))
        kw = {k: 2 for k in kinds if k == "indent"}
        for payload in ({"a": 1}, [1, 2, 3]):          # both payload types in use
            if kind == "handle":
                with open(f, "w") as fh:
                    P.dump(payload, fh, **kw)
            else:
                P.dump(payload, f, **kw)
            got = json.load(open(f))
            assert isinstance(got, type(payload)), (relpath, lineno, type(got))
            if isinstance(got, dict):
                assert "_provenance" in got, "%s:%d lost its provenance" % (relpath, lineno)
        seen.add((kind, tuple(kinds)))
        print("  ok  %-38s:%-4d  dest=%-6s kwargs=%s" % (relpath, lineno, kind, kinds or "-"))
    print("\ndistinct call shapes covered: %s" % sorted(seen))
    print("PASS -- every real PROV.dump call site works with both payload types")
    return 0


if __name__ == "__main__":
    sys.exit(main())

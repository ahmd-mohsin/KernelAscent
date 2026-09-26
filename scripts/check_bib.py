#!/usr/bin/env python3
"""Bibliography hygiene: nothing cited that does not exist, nothing listed that is not cited,
and nothing unverified allowed to pass silently.

A fabricated citation is the one error in this paper that no other gate would catch and that no
reviewer would forgive. The repo's own citations note says so and then had no mechanism behind
it. This is the mechanism.

Three checks:
  * every \\cite key resolves to an entry in refs.bib          -- a broken cite prints "[?]"
  * every entry in refs.bib is cited somewhere                 -- an uncited entry pads the list
  * every entry declares `verified`, and the ones marked `no`  -- these were written from
    are listed loudly rather than counted as fine                 knowledge and not checked
                                                                  against the publisher record

Exit 1 on the first two. The third is a WARNING by default and an error under --strict, because
an unverified entry is fine while drafting and unacceptable at submission.

  python3 scripts/check_bib.py            # drafting
  python3 scripts/check_bib.py --strict   # before submitting
"""
import argparse, glob, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIB = os.path.join(ROOT, "paper", "refs.bib")


def entries():
    """key -> dict of fields, for every @type{key, ...} in the bib."""
    try:
        txt = open(BIB, encoding="utf-8").read()
    except FileNotFoundError:
        return None
    out = {}
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", txt):
        key = m.group(2)
        # field scan to the next @entry or EOF; good enough for a flat bib and avoids a parser
        nxt = txt.find("\n@", m.end())
        body = txt[m.end(): nxt if nxt != -1 else len(txt)]
        # one capture group -> findall returns strings, not pairs; a set of field NAMES is all
        # this needs, and dict() on that list raised rather than silently doing something odd.
        fields = set(re.findall(r"(\w+)\s*=\s*\{", body))
        ver = re.search(r"verified\s*=\s*\{([^}]*)\}", body)
        out[key] = {"type": m.group(1), "fields": fields,
                    "verified": (ver.group(1).strip() if ver else None)}
    return out


def cited():
    """key -> [files], for every \\cite-family command across the paper sources."""
    out = {}
    for path in sorted(glob.glob(os.path.join(ROOT, "paper", "*.tex"))):
        txt = open(path, encoding="utf-8").read()
        # strip comments so a commented-out \cite does not count as a citation
        txt = re.sub(r"(?<!\\)%.*", "", txt)
        for m in re.finditer(r"\\(?:cite|citep|citet|citealp)\*?(?:\[[^\]]*\])*\{([^}]*)\}", txt):
            for key in (k.strip() for k in m.group(1).split(",")):
                if key:
                    out.setdefault(key, []).append(os.path.basename(path))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true",
                    help="fail on entries whose bibliographic fields are unverified")
    a = ap.parse_args()

    ents = entries()
    if ents is None:
        print("no paper/refs.bib -- nothing to check")
        return 0
    cites = cited()
    fails, warns = [], []

    missing = sorted(k for k in cites if k not in ents)
    for k in missing:
        fails.append("cited but absent from refs.bib: %s (in %s)"
                     % (k, ", ".join(sorted(set(cites[k])))))

    uncited = sorted(k for k in ents if k not in cites)
    for k in uncited:
        fails.append("in refs.bib but never cited: %s" % k)

    # Two provenances count as checked, and they are different kinds of evidence:
    #   spec -- transcribed from the project's own reference list, which carries author/venue
    #   web  -- confirmed this session against the publisher or arXiv record
    # Anything else was written from memory and is the only category that can be fabricated.
    OK = {"spec", "web"}
    unverified = sorted(k for k, v in ents.items() if v["verified"] not in OK)
    for k in unverified:
        state = ents[k]["verified"] or "absent"
        (fails if a.strict else warns).append(
            "written from memory, not checked against any record: %s (verified=%s)" % (k, state))

    # An entry that names no author and no organization cannot be looked up by a reader.
    for k, v in sorted(ents.items()):
        if not (v["fields"] & {"author", "organization", "howpublished"}):
            fails.append("no author, organization or howpublished: %s" % k)

    print("=" * 84)
    by = {}
    for v in ents.values():
        by[v["verified"] or "absent"] = by.get(v["verified"] or "absent", 0) + 1
    print("BIBLIOGRAPHY  %d entries, %d cited, %d distinct keys used"
          % (len(ents), len(ents) - len(uncited), len(cites)))
    print("              provenance: %s"
          % ", ".join("%s=%d" % kv for kv in sorted(by.items())))
    print("=" * 84)
    for w in warns:
        print("  [warn] %s" % w)
    for f in fails:
        print("  [FAIL] %s" % f)
    if not fails:
        print("  none -- every citation resolves and every entry is used")
        if warns and not a.strict:
            print("\n  %d entr%s still need their fields confirmed before submission."
                  % (len(warns), "y" if len(warns) == 1 else "ies"))
            print("  Run with --strict to make that a failure.")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

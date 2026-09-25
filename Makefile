# KernelAscent paper figures + result tables
.PHONY: figures audit stats paper site all

# Recompute the clustered (trajectory-level) statistics from the raw trajectories.
# These feed results_auto.tex, so they run before the paper is regenerated.
stats:
	python3 scripts/equivalence_tost.py --dir data/trajectories --out docs/data/compounding_tost.json
	python3 scripts/search_vs_train.py --dir data/trajectories
	python3 scripts/mech_analysis.py
	python3 scripts/probe_appendix.py --tex paper/probe_appendix.tex --json docs/data/probe_rigor.json

# Cross-artifact consistency gate: every number that appears in more than one place must agree.
# Non-zero exit fails the build on purpose -- see scripts/consistency_audit.py.
audit:
	python3 scripts/consistency_audit.py
	python3 tests/test_audit_gates.py
	python3 tests/test_provenance_shim.py
	python3 scripts/check_resume.py --strict
	python3 tests/test_probe_imports.py
	python3 scripts/scan_contamination.py

figures: stats
	python3 scripts/make_paper_figures.py
	python3 scripts/make_causality_figures.py
	python3 scripts/make_gallery_figures.py
	python3 scripts/make_intervene_figures.py
	python3 scripts/make_results_tex.py
	python3 scripts/make_kernel_results_tex.py
	python3 scripts/build_site_headline.py
	$(MAKE) audit
	cd paper && (tectonic figures.tex || (pdflatex -interaction=nonstopmode figures.tex && pdflatex -interaction=nonstopmode figures.tex))

# The consolidated report. `figures` builds the figure atlas, which is a different document;
# until this target existed the one thing a reader is handed was the only artifact the build
# never produced, and its tables could be regenerated without anyone noticing the paper no
# longer compiled.
paper:
	python3 scripts/make_kernel_results_tex.py
	cd paper && pdflatex -interaction=nonstopmode kernelascent.tex >/dev/null
	cd paper && pdflatex -interaction=nonstopmode kernelascent.tex | \
	  grep -E "Output written|^!|Reference .* undefined" || true

# The website's headline numbers, re-derived from the same artifacts through the same module
# the paper's tables use. docs/index.html reads the result at page load and types none of it.
site:
	python3 scripts/build_site_headline.py
	python3 scripts/consistency_audit.py

all: figures paper site

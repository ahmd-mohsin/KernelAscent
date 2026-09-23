# KernelAscent paper figures + result tables
.PHONY: figures audit stats

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

figures: stats
	python3 scripts/make_paper_figures.py
	python3 scripts/make_causality_figures.py
	python3 scripts/make_gallery_figures.py
	python3 scripts/make_intervene_figures.py
	python3 scripts/make_results_tex.py
	$(MAKE) audit
	cd paper && (tectonic figures.tex || (pdflatex -interaction=nonstopmode figures.tex && pdflatex -interaction=nonstopmode figures.tex))

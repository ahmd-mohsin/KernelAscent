# KernelAscent paper figures
.PHONY: figures
figures:
	python3 scripts/make_paper_figures.py
	cd paper && (tectonic figures.tex || (pdflatex -interaction=nonstopmode figures.tex && pdflatex -interaction=nonstopmode figures.tex))

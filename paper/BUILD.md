# Building the paper

Needs TeX Live (IEEEtran, newtx, biblatex, biber, tikz). From this folder:

    latexmk -pdf main.tex

Tables (`tables/*.tex`), figures (`figures/*.pdf`) and `sections/abstract.tex` are generated from the training run files by
`train/src/report.py` in the full project; they are included here so the paper builds on its own. `main.bbl` is the compiled
bibliography (rebuilt automatically by latexmk from `refs.bib`). The class is `[conference]` (author-block layout copied from the
sem5 deeplearning paper); change it to `[journal]` for a journal template.

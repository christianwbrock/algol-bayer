
all:: fast_extraction_poster.pdf
all:: fast_extraction.pdf

clean::
	$(RM) *.aux *.bbl  *.log *.blg *.out *.synctex.gz *.toc *.nav *.snm

mr_proper: clean
	$(RM) fast_extraction_poster.pdf fast_extraction.pdf


%.pdf: %.tex %.aux %.blg
	pdflatex $<

%.aux: %.tex
	pdflatex $<

%.blg: %.aux
	bibtex $<


#!/bin/bash

rm -r release
mkdir release
cp p1_pandas.py release/q1_q2.py
cp ent_sweep.py release
cp TechnoInfoCom.pdf release/P1_Champailler_Gomes.pdf

# pdflatex flight.tex
# cp flight.pdf release/report.pdf
zip -r project1.zip release --exclude \*.pyc
ls -l p*zip

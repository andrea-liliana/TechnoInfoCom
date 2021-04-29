#!/bin/bash

NAME=s912550s198387

rm -f -r release
mkdir release
cp p2_channel.py p2_huffman.py p2_LZ77.py p2_main.py p2_online_lz.py p2_utils.py hamming.py release
cp genome.txt sound.wav release
cp TechnoInfoCom.pdf release/${NAME}.pdf

# pdflatex flight.tex
# cp flight.pdf release/report.pdf
rm ${NAME}.zip
zip -r ${NAME}.zip release --exclude \*.pyc --exclude \*.dot
ls -l ${NAME}.zip
unzip -l ${NAME}.zip

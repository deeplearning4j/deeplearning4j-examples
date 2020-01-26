#!/bin/bash

# usage: 
# ./word2vec-download300model.sh output-file

set -o errexit
set -o pipefail

CODE=$( wget --save-cookies cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=0B7XkCwpI5KDYNlNUTTlSS21pQmM' -O- | perl -0p -e 's/.*confirm=(\S+)\;id=.*/$1\n/s' )

mkdir -p $1
OUT_F=$1/'GoogleNews-vectors-negative300.bin.gz'

wget --load-cookies cookies.txt 'https://docs.google.com/uc?export=download&confirm='$CODE'&id=0B7XkCwpI5KDYNlNUTTlSS21pQmM' -O $OUT_F

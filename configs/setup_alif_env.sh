#!/bin/bash
# ALIF (third_party/TASER) runs on the SMACK environment (torch, librosa, python-Levenshtein, inflect)
# plus unidecode for Tacotron2's text cleaners. Create the smack env first (configs/smack.yml).
set -e

conda create -y --name alif --clone smack
conda run -n alif pip install unidecode

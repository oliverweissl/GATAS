"""Classify the normalized texts of all intent-task results once and cache them for results_loader.

The intent of a transcription is measured on normalized text (lowercase, no punctuation), so that the casing and
punctuation of an ASR model (e.g. the uppercase output of wav2vec2) do not count as intent changes. The reference
intent is measured on the normalized clean text in the same way. Run from analysis/: python normalize_intents.py
"""
import json
import os
import sys
from glob import glob

sys.path.insert(0, "..")
from results_loader import INTENT_CACHE, normalize_text
from src.intent import classify_intents

texts = set()
for path in glob("../outputs/results/*/commands/*/**/evaluation.json", recursive=True):
    with open(path) as f:
        e = json.load(f)
    texts |= {normalize_text(e.get(k)) for k in ("gt_text", "original_transcription", "transcription")}
texts = sorted(texts)
with open(INTENT_CACHE, "w") as f:
    json.dump(dict(zip(texts, classify_intents(texts))), f, indent=0)
print(f"{len(texts)} texts -> {os.path.abspath(INTENT_CACHE)}")

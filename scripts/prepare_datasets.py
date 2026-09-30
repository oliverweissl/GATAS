"""
Build the seed-sentence lists for the LibriSpeech and voice-command datasets (Harvard sentences ship in
src/data/harvard_sentences.py). Only transcripts are used: GATAS and all baselines synthesize the reference
speech with TTS. The selected sentences are committed in src/data/seed_sentences/, this script documents
and reproduces the selection:

    python scripts/prepare_datasets.py

  librispeech: LibriSpeech test-clean transcripts (openslr/librispeech_asr on the Hugging Face Hub),
               8-20 words, random sample of 100 (seed 0).
  commands:    SLURP test-set voice-assistant commands (github.com/pswietojanski/slurp),
               >= 4 words and >= 2 content words, unique, random sample of 100 (seed 0).
"""
import os
import re
import sys
import json
import random
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.datasets import SEED_SENTENCE_DIR
from src.metrics import content_lemmas

NUM_SENTENCES = 100
SEED = 0

LIBRISPEECH_ROWS = "https://datasets-server.huggingface.co/rows?dataset=openslr/librispeech_asr&config=clean&split=test&offset={offset}&length=100"
SLURP_TEST = "https://raw.githubusercontent.com/pswietojanski/slurp/master/dataset/slurp/test.jsonl"


def _get(url: str) -> str:
    with urllib.request.urlopen(url) as response:
        return response.read().decode("utf-8")


def _sentence_case(text: str) -> str:
    text = " ".join(text.lower().split())
    text = re.sub(r"\bi\b", "I", text)
    return text[0].upper() + text[1:] + "."


def librispeech() -> list[str]:
    texts, offset = [], 0
    while True:
        rows = json.loads(_get(LIBRISPEECH_ROWS.format(offset=offset)))["rows"]
        if not rows:
            break
        texts += [row["row"]["text"] for row in rows]
        offset += len(rows)
    candidates = sorted({t for t in texts if 8 <= len(t.split()) <= 20})
    return [_sentence_case(t) for t in random.Random(SEED).sample(candidates, NUM_SENTENCES)]


def commands() -> list[str]:
    texts = [json.loads(line)["sentence"] for line in _get(SLURP_TEST).splitlines() if line.strip()]
    candidates = sorted({t.strip() for t in texts if len(t.split()) >= 4 and len(content_lemmas(t)) >= 2})
    return [_sentence_case(t) for t in random.Random(SEED).sample(candidates, NUM_SENTENCES)]


def main():
    os.makedirs(SEED_SENTENCE_DIR, exist_ok=True)
    for name, build in (("librispeech", librispeech), ("commands", commands)):
        sentences = build()
        path = os.path.join(SEED_SENTENCE_DIR, f"{name}.txt")
        with open(path, "w") as f:
            f.write("\n".join(sentences) + "\n")
        print(f"{name}: {len(sentences)} sentences -> {path}")


if __name__ == "__main__":
    main()

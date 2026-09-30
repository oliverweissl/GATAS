"""
Seed-sentence datasets. Sentence IDs are 1-based positions in the list.

  harvard:     Harvard sentences (IEEE 1969), src/data/harvard_sentences.py
  librispeech: 100 LibriSpeech test-clean transcripts (scripts/prepare_datasets.py)
  commands:    100 SLURP voice-assistant commands (scripts/prepare_datasets.py)
"""
import os
from functools import lru_cache

from .harvard_sentences import HARVARD_SENTENCES

DATASETS = ("harvard", "librispeech", "commands")
SEED_SENTENCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seed_sentences")


@lru_cache(maxsize=None)
def load_sentences(dataset: str) -> tuple:
    if dataset == "harvard":
        return tuple(HARVARD_SENTENCES)
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose one of: {', '.join(DATASETS)}")
    with open(os.path.join(SEED_SENTENCE_DIR, f"{dataset}.txt")) as f:
        return tuple(line.strip() for line in f if line.strip())


def get_sentence(dataset: str, sentence_id: int) -> str:
    """1-based sentence ID."""
    return load_sentences(dataset)[sentence_id - 1]

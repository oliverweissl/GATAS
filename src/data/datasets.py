"""
Seed-sentence datasets. Sentence IDs are 1-based positions in the list.

  harvard:  Harvard sentences (IEEE 1969), src/data/harvard_sentences.py
  commands: 100 SLURP voice-assistant commands (scripts/prepare_datasets.py)
"""
import os
from functools import lru_cache

from .harvard_sentences import HARVARD_SENTENCES

DATASETS = ("harvard", "commands")
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


def _load_lines(name: str):
    path = os.path.join(SEED_SENTENCE_DIR, f"{name}.txt")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return tuple(line.strip() for line in f if line.strip())


@lru_cache(maxsize=None)
def load_intents(dataset: str):
    """Reference intent per sentence (same order as load_sentences), or None if the dataset has no intents."""
    return _load_lines(f"{dataset}.intents")


@lru_cache(maxsize=None)
def load_target_intents(dataset: str):
    """Target intent per sentence for the intent matching attacks (second most likely intent), or None."""
    return _load_lines(f"{dataset}.target_intents")


@lru_cache(maxsize=None)
def load_intent_rewrites(dataset: str):
    """LLM rewrite per sentence whose intent is the target intent (scripts/generate_intent_rewrites.py), or None."""
    return _load_lines(f"{dataset}.intent_rewrites")

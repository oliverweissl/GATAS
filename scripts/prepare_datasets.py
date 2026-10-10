"""
Build the seed-sentence lists for the voice-command dataset (Harvard sentences ship in
src/data/harvard_sentences.py). Only transcripts are used: GATAS and all baselines synthesize the reference
speech with TTS. The selected sentences are committed in src/data/seed_sentences/, this script documents
and reproduces the selection:

    python scripts/prepare_datasets.py

  commands: SLURP test-set voice-assistant commands (github.com/pswietojanski/slurp),
            >= 4 words and >= 2 content words, unique, one gold intent, random sample of 100 (seed 0).
            commands.intents.txt: reference intent per sentence = most likely intent of the intent classifier
            (src/intent.py) on the clean text; commands.target_intents.txt: its most likely well-formed intent of a different
            scenario (target of the intent matching attacks); commands.gold_intents.txt: SLURP gold intents (documentation only).
"""
import os
import re
import sys
import json
import random
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.datasets import SEED_SENTENCE_DIR
from src.intent import intent_labels, intent_probabilities
from src.metrics import content_lemmas

NUM_SENTENCES = 100
SEED = 0

SLURP_TEST = "https://raw.githubusercontent.com/pswietojanski/slurp/master/dataset/slurp/test.jsonl"


def _get(url: str) -> str:
    with urllib.request.urlopen(url) as response:
        return response.read().decode("utf-8")


def _sentence_case(text: str) -> str:
    text = " ".join(text.lower().split())
    text = re.sub(r"\bi\b", "I", text)
    return text[0].upper() + text[1:] + "."


def commands() -> tuple[list[str], list[str]]:
    """Sentences and their gold SLURP intents (scenario_action, e.g. alarm_query), same order."""
    rows = [json.loads(line) for line in _get(SLURP_TEST).splitlines() if line.strip()]
    intents = {}
    for row in rows:
        intents.setdefault(row["sentence"].strip(), set()).add(row["intent"])
    candidates = sorted(t for t in intents if len(t.split()) >= 4 and len(content_lemmas(t)) >= 2)
    candidates = [t for t in candidates if len(intents[t]) == 1]  # a sentence with several gold intents is ambiguous
    chosen = random.Random(SEED).sample(candidates, NUM_SENTENCES)
    return [_sentence_case(t) for t in chosen], [next(iter(intents[t])) for t in chosen]


def main():
    os.makedirs(SEED_SENTENCE_DIR, exist_ok=True)
    sentences, gold = commands()
    # Reference intent of a sentence: the classifier's most likely intent on the clean text, so the classifier agrees
    # with the reference on every clean sentence by construction; SLURP's gold labels (ambiguous for ~13 % of the
    # sentences) are kept for documentation only. Target intent of the intent matching attacks: the classifier's most
    # likely intent on the clean text after the reference that is well-formed (scenario_action; SLURP also contains
    # scenario-less twins such as `query`, `set`, `radio`) and in a different scenario than the reference, i.e. the
    # nearest decision boundary that is a real change of intent. (The plain second most likely intent is such a
    # twin or stays in the scenario for 36 resp. 32 of the 100 sentences.)
    labels = intent_labels()
    ranked = intent_probabilities(sentences).argsort(axis=1)[:, ::-1]
    reference = [labels[row[0]] for row in ranked]

    def scenario(label):
        return label.split("_")[0]

    target = [next(labels[i] for i in row if "_" in labels[i] and scenario(labels[i]) != scenario(ref))
              for row, ref in zip(ranked, reference)]
    print(f"classifier vs SLURP gold: {sum(r == g for r, g in zip(reference, gold))}/{len(gold)} agree")
    for name, lines in (("commands", sentences), ("commands.intents", reference), ("commands.target_intents", target),
                        ("commands.gold_intents", gold)):
        path = os.path.join(SEED_SENTENCE_DIR, f"{name}.txt")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"{name}: {len(lines)} lines -> {path}")


if __name__ == "__main__":
    main()

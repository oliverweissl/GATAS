"""
Rewrite every seed sentence with a local LLM (Ollama) so that its intent becomes the target intent, with as few word
changes as possible. The rewrites are the targets of the GATAS latent interpolation variant (GATAS_llm); they are cached in
src/data/seed_sentences/<dataset>.intent_rewrites.txt (one per line) so that every run uses the identical target text.

Each candidate is verified with the intent classifier (target intent must be the predicted intent); the best of
`--tries` samples is kept: target predicted first, then the fewest changed words.

Run from project root (main env): python scripts/generate_intent_rewrites.py --dataset commands
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import difflib
import json
import re
import urllib.request

from src.data.datasets import SEED_SENTENCE_DIR, load_sentences, load_intents, load_target_intents
from src.intent import intent_labels, intent_probabilities

PROMPT = """You edit voice-assistant commands (SLURP dataset).
Rewrite the command below with as FEW word changes as possible so that it expresses the intent "{target}" (scenario "{scenario}", action "{action}") instead of the current intent "{current}".
Keep the rest of the wording, keep it a natural spoken command, and do not add explanations.

Command: {sentence}
Rewritten command (output only the sentence):"""


def ollama(host: str, model: str, prompt: str, temperature: float, seed: int) -> str:
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "options": {"temperature": temperature, "seed": seed, "num_predict": 60}}).encode()
    request = urllib.request.Request(f"{host}/api/generate", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        text = json.load(response)["response"].strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    return text.splitlines()[0].strip().strip('"').strip() if text else ""


def changed_words(a: str, b: str) -> int:
    a, b = a.lower().split(), b.lower().split()
    return sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes() if tag != "equal")


def main():
    parser = argparse.ArgumentParser(description="LLM intent rewrites of the seed sentences")
    parser.add_argument("--dataset", type=str, default="commands")
    parser.add_argument("--model", type=str, default="qwen2.5:latest")
    parser.add_argument("--host", type=str, default="http://localhost:11434")
    parser.add_argument("--tries", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu", help="device of the intent classifier")
    args = parser.parse_args()

    sentences, currents, targets = load_sentences(args.dataset), load_intents(args.dataset), load_target_intents(args.dataset)
    labels = intent_labels(args.device)
    rewrites, hits = [], 0
    for i, (sentence, current, target) in enumerate(zip(sentences, currents, targets), 1):
        scenario, _, action = target.partition("_")
        prompt = PROMPT.format(target=target, scenario=scenario, action=action, current=current, sentence=sentence)
        candidates = [c for c in (ollama(args.host, args.model, prompt, 0.2 + 0.1 * t, t) for t in range(args.tries)) if c and c != sentence]
        if not candidates:
            candidates = [sentence]
        probabilities = intent_probabilities(candidates, args.device)
        column = labels.index(target)
        best = min(range(len(candidates)), key=lambda k: (labels[probabilities[k].argmax()] != target,
                                                          changed_words(sentence, candidates[k]), -probabilities[k][column]))
        hit = labels[probabilities[best].argmax()] == target
        hits += hit
        rewrites.append(candidates[best])
        print(f"[{i:3d}] {'OK ' if hit else 'MISS'} P={probabilities[best][column]:.2f} {current}->{target}: {sentence!r} => {candidates[best]!r}", flush=True)

    with open(os.path.join(SEED_SENTENCE_DIR, f"{args.dataset}.intent_rewrites.txt"), "w") as f:
        f.write("\n".join(rewrites) + "\n")
    print(f"[Done] {hits}/{len(rewrites)} rewrites are classified as the target intent")


if __name__ == "__main__":
    main()

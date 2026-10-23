"""
Re-scores the saved ASR-task results (harvard) with two semantic-change judges, without new ASR queries:
  nli  bidirectional entailment of a pretrained NLI cross-encoder; changed = not (gt => hyp and hyp => gt)
  llm  local ollama model, binary verdict "same meaning?" at temperature 0
Pairs (gt_text, transcription) are deduplicated and cached in --cache, so a rerun only scores new pairs.
Run from project root: CUDA_VISIBLE_DEVICES=1 python scripts/semantic_rescore.py [--no_llm] [--no_nli]
"""
import argparse
import json
import os
import urllib.request
from glob import glob

NLI_MODEL = "cross-encoder/nli-deberta-v3-large"
PROMPT = (
    "You compare the output of a speech recognition system with the reference transcript.\n"
    "Decide whether the recognised text still conveys the same meaning as the reference. Ignore capitalisation, punctuation and "
    "spelling of numbers. Different words, a missing or added content word, or a changed word that alters who/what/when/how "
    "count as a change of meaning; a missing or swapped function word that leaves the meaning intact does not.\n\n"
    'Reference: "{a}"\nRecognised: "{b}"\n\nAnswer with JSON: {{"same_meaning": true|false}}'
)

PROMPT3 = (
    "You compare the output of a speech recognition system with the reference transcript.\n"
    "Classify the recognised text into exactly one class:\n"
    '- "preserved": it conveys the same meaning as the reference (ignore capitalisation, punctuation, spelling of numbers, '
    "harmless function-word differences).\n"
    '- "changed": it is a coherent sentence or phrase, but its meaning differs from the reference.\n'
    '- "garbled": it is not coherent (nonsense words or word salad, a native speaker could not recover a sentence) or empty.\n\n'
    'Reference: "{a}"\nRecognised: "{b}"\n\nAnswer with JSON: {{"verdict": "preserved"|"changed"|"garbled"}}'
)


def collect_pairs(results_root, models):
    pairs = {}
    for path in glob(os.path.join(results_root, "*", "harvard", "*", "**", "evaluation.json"), recursive=True):
        parts = os.path.relpath(path, results_root).split(os.sep)
        if not any(parts[2].startswith(m + "_") for m in models):
            continue
        with open(path) as f:
            e = json.load(f)
        pairs[(e["gt_text"], e["transcription"] or "")] = None
    return list(pairs)


def key(pair):
    return json.dumps(pair)


def run_nli(pairs, cache, device):
    from sentence_transformers import CrossEncoder
    import numpy as np

    model = CrossEncoder(NLI_MODEL, device=device)
    labels = [model.model.config.id2label[i].lower() for i in range(model.model.config.num_labels)]
    ent = labels.index("entailment")
    todo = [p for p in pairs if "nli_changed" not in cache.get(key(p), {})]
    print(f"NLI: {len(todo)} pairs, labels {labels}")
    for i in range(0, len(todo), 64):
        batch = todo[i:i + 64]
        fwd = model.predict([(a, b) for a, b in batch], apply_softmax=True, batch_size=32)
        bwd = model.predict([(b, a) for a, b in batch], apply_softmax=True, batch_size=32)
        for p, f, b in zip(batch, fwd, bwd):
            cache.setdefault(key(p), {}).update({
                "nli_fwd": float(f[ent]), "nli_bwd": float(b[ent]),
                "nli_changed": not (int(np.argmax(f)) == ent and int(np.argmax(b)) == ent)})


def run_llm(pairs, cache, model, url):
    todo = [p for p in pairs if f"llm_changed:{model}" not in cache.get(key(p), {})]
    print(f"LLM ({model}): {len(todo)} pairs")
    for i, (a, b) in enumerate(todo, 1):
        body = {"model": model, "stream": False, "think": False, "options": {"temperature": 0, "num_predict": 200},
                "messages": [{"role": "user", "content": PROMPT.format(a=a, b=b)}],
                "format": {"type": "object", "properties": {"same_meaning": {"type": "boolean"}}, "required": ["same_meaning"]}}
        req = urllib.request.Request(url + "/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            out = json.loads(json.load(r)["message"]["content"])
        cache.setdefault(key((a, b)), {})[f"llm_changed:{model}"] = not out["same_meaning"]
        if i % 100 == 0:
            print(f"  {i}/{len(todo)}")


def run_llm3(pairs, cache, model, url):
    """Three-way verdict (preserved / changed / garbled), stored as llm3:<model>."""
    todo = [p for p in pairs if f"llm3:{model}" not in cache.get(key(p), {})]
    print(f"LLM3 ({model}): {len(todo)} pairs")
    schema = {"type": "object", "properties": {"verdict": {"type": "string", "enum": ["preserved", "changed", "garbled"]}},
              "required": ["verdict"]}
    for i, (a, b) in enumerate(todo, 1):
        body = {"model": model, "stream": False, "think": False, "options": {"temperature": 0, "num_predict": 200},
                "messages": [{"role": "user", "content": PROMPT3.format(a=a, b=b)}], "format": schema}
        req = urllib.request.Request(url + "/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            cache.setdefault(key((a, b)), {})[f"llm3:{model}"] = json.loads(json.load(r)["message"]["content"])["verdict"]
        if i % 100 == 0:
            print(f"  {i}/{len(todo)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", default="outputs/results")
    ap.add_argument("--models", nargs="+", default=["whisper-tiny", "wav2vec2-large"])
    ap.add_argument("--cache", default="outputs/semantic_scores.json")
    ap.add_argument("--llm", default="gemma4:31b")
    ap.add_argument("--ollama_url", default="http://localhost:11434")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no_nli", action="store_true")
    ap.add_argument("--no_llm", action="store_true")
    ap.add_argument("--llm3", action="store_true", help="three-way verdict instead of the binary one")
    a = ap.parse_args()

    cache = json.load(open(a.cache)) if os.path.exists(a.cache) else {}
    pairs = collect_pairs(a.results_root, a.models)
    print(f"{len(pairs)} unique (gt, transcription) pairs")
    if not a.no_nli:
        run_nli(pairs, cache, a.device)
        json.dump(cache, open(a.cache, "w"))
    if not a.no_llm:
        try:
            (run_llm3 if a.llm3 else run_llm)(pairs, cache, a.llm, a.ollama_url)
        finally:
            json.dump(cache, open(a.cache, "w"))
    json.dump(cache, open(a.cache, "w"))
    print("saved", a.cache)

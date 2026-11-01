"""
Classifies the saved ASR-task results (harvard) as preserved / changed / garbled in meaning with a local ollama LLM at
temperature 0, without new ASR queries (semantic outcome, RQ1). Pairs (gt_text, transcription) are deduplicated and cached
in --cache, so a rerun only scores new pairs. Run from project root: python scripts/semantic_rescore.py
"""
import argparse
import json
import os
import urllib.request
from glob import glob

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
    a = ap.parse_args()

    cache = json.load(open(a.cache)) if os.path.exists(a.cache) else {}
    pairs = collect_pairs(a.results_root, a.models)
    print(f"{len(pairs)} unique (gt, transcription) pairs")
    try:
        run_llm3(pairs, cache, a.llm, a.ollama_url)
    finally:
        json.dump(cache, open(a.cache, "w"))
    print("saved", a.cache)

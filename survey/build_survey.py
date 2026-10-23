"""Build the realism survey for GitHub Pages: loudness-normalized audios under anonymous ids in docs/audio, docs/items.json (public)
and survey/private/key.json (item id -> condition, git-ignored).

Four cells (SUT x task). Conditions per cell: GATAS (GATAS_llm on intent), the two best baselines (survey/select_baselines.py) and the
clean audio. Each condition has its own pool: sentences the SUT handles correctly on clean input and for which the method found a test
case, at most MAX_POOL. Run from the project root: python survey/build_survey.py
"""
import json
import os
import random
import shutil
import subprocess
import sys

sys.path.insert(0, "analysis")
sys.path.insert(0, "survey")
from results_loader import load_results, asr_found
from select_baselines import select, ASR, INTENT

OUT = "docs"
PRIVATE = "survey/private"
LOUDNESS = "I=-23:TP=-2:LRA=11"  # EBU R128


def normalize(src, dst):
    """Two-pass EBU R128 loudness normalization (linear gain, no dynamic processing) to 16 kHz mono 16 bit wav."""
    probe = subprocess.run(["ffmpeg", "-hide_banner", "-i", src, "-af", f"loudnorm={LOUDNESS}:print_format=json", "-f", "null", "-"],
                           capture_output=True, text=True).stderr
    m = json.loads(probe[probe.rindex("{"):])
    f = (f"loudnorm={LOUDNESS}:measured_I={m['input_i']}:measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}"
         f":measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-af", f, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", dst], check=True)


BASELINES = select()  # the two best baselines of RQ1 by reciprocal rank fusion, see survey/select_baselines.py
print("baselines:", BASELINES)
ASR_FOLDER = ASR  # SMACK: the native configuration, which has the better rank sum (Changed, UTMOS) than the untargeted adaptation
CELLS = {  # id: (label, dataset, SUT, {condition: result folder})
    "wt-asr": ("Whisper-tiny, ASR", "harvard", "whisper-tiny", {"ours": "GATAS", **{b: ASR_FOLDER[b] for b in BASELINES}}),
    "w2v-asr": ("wav2vec2-large, ASR", "harvard", "wav2vec2-large", {"ours": "GATAS", **{b: ASR_FOLDER[b] for b in BASELINES}}),
    "wt-intent": ("Whisper-tiny, intent", "commands", "whisper-tiny", {"ours": "GATAS_llm", **{b: INTENT[b] for b in BASELINES}}),
    "w2v-intent": ("wav2vec2-large, intent", "commands", "wav2vec2-large", {"ours": "GATAS_llm", **{b: INTENT[b] for b in BASELINES}}),
}
MAX_POOL = 40


def silent(wav):
    import soundfile as sf
    return float(abs(sf.read(wav)[0]).max()) < 1e-4  # cannot be rated nor loudness-normalized


def found(r, dataset):
    return asr_found(r) if dataset == "harvard" else bool(r["intent"] and r["intent"]["original_correct"] and r["intent"]["changed"])


def shorten(text, n=100):
    text = text.strip()
    return "" if not text else text if len(text) <= n else text[:n].rstrip() + " ..."


rng = random.Random(0)
shutil.rmtree(f"{OUT}/audio", ignore_errors=True)
os.makedirs(f"{OUT}/audio"); os.makedirs(PRIVATE, exist_ok=True)
cells, key = {}, {}
for cell, (label, dataset, sut, folders) in CELLS.items():
    records = load_results("outputs/results", dataset=dataset, asr_model=sut, methods=tuple(folders.values()), exclude_original_fails=False)
    by_cond = {c: {r["sentence_id"]: r for r in records[f] if found(r, dataset) and (dataset == "commands" or r["original_passes"])}
               for c, f in folders.items()}
    # every condition has its own pool: the sentences for which the method produced a test case (the clean audio is always available)
    pools = {c: sorted(sid for sid in d if not silent(d[sid]["adv_wav"])) for c, d in by_cond.items()}
    pools = {c: sorted(rng.sample(p, MAX_POOL)) if len(p) > MAX_POOL else p for c, p in pools.items()}
    union = sorted(set().union(*pools.values()))
    pools["clean"] = sorted(rng.sample(union, MAX_POOL)) if len(union) > MAX_POOL else union
    print(f"{cell}: pools {{{', '.join(f'{c}: {len(p)}' for c, p in pools.items())}}}")
    codes = list("ABCD"); rng.shuffle(codes)  # opaque condition codes: the client balances conditions without learning which is which
    code_of = dict(zip(("clean",) + tuple(folders), codes))
    items = []
    for cond, pool in pools.items():
        for sid in pool:
            ref = by_cond["ours"].get(sid) or next(d[sid] for d in by_cond.values() if sid in d)
            wav, transcription = (ref["gt_wav"], ref["original_transcription"]) if cond == "clean" else (by_cond[cond][sid]["adv_wav"], by_cond[cond][sid]["transcription"])
            item_id = f"i{rng.getrandbits(40):010x}"
            normalize(wav, f"{OUT}/audio/{item_id}.wav")
            items.append({"id": item_id, "code": code_of[cond], "sentence": sid, "gt_text": ref["gt_text"], "transcription": shorten(transcription)})
            key[item_id] = {"cell": cell, "sentence": sid, "condition": cond, "source": os.path.relpath(wav)}
    cells[cell] = {"label": label, "items": items}
json.dump({"cells": cells, "per_condition": 5}, open(f"{OUT}/items.json", "w"), indent=1)
json.dump(key, open(f"{PRIVATE}/key.json", "w"), indent=1)
print(f"{len(key)} items")

"""
Build the realism survey (ASR task, Whisper-tiny) for GitHub Pages: writes the loudness-normalized audios under anonymous ids to
docs/audio and docs/items.json (public), and survey/private/key.json (item id -> method/sentence, git-ignored, never published).

Conditions: the five methods of the ASR comparison (for SMACK the variant with the better RQ1 effectiveness, the untargeted
adaptation) and the clean reference audio as control. The item pool are the sentences whose clean audio is transcribed
correctly and for which every method found a test case, so each sentence exists in all six conditions.
Run from the project root: python survey/build_survey.py
"""
import json
import os
import random
import shutil
import subprocess
import sys

sys.path.insert(0, "analysis")
from results_loader import load_results, asr_found

METHODS = ("GATAS", "Waveform", "SMACK_untargeted", "ALIF", "PGD")
OUT = "docs"
PRIVATE = "survey/private"
LOUDNESS = "I=-23:TP=-2:LRA=11"  # EBU R128 target for every audio, so no clip is much louder or quieter than the others


def normalize(src, dst):
    """Two-pass EBU R128 loudness normalization (linear gain, no dynamic processing) to 16 kHz mono 16 bit wav."""
    probe = subprocess.run(["ffmpeg", "-hide_banner", "-i", src, "-af", f"loudnorm={LOUDNESS}:print_format=json", "-f", "null", "-"],
                           capture_output=True, text=True).stderr
    m = json.loads(probe[probe.rindex("{"):])
    f = (f"loudnorm={LOUDNESS}:measured_I={m['input_i']}:measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}"
         f":measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-af", f, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", dst], check=True)


records = load_results("outputs/results", dataset="harvard", asr_model="whisper-tiny", methods=METHODS, exclude_original_fails=True)
by_method = {m: {r["sentence_id"]: r for r in rs if asr_found(r)} for m, rs in records.items()}
pool = sorted(set.intersection(*(set(d) for d in by_method.values())))
print(f"{len(pool)} sentences with a test case from every method")



def shorten(text, n=100):
    text = text.strip()
    return "" if not text else text if len(text) <= n else text[:n].rstrip() + " ..."


rng = random.Random(0)
codes = list("ABCDEF"); rng.shuffle(codes)  # opaque condition codes: the client balances conditions without learning which is which
code_of = dict(zip(("clean",) + METHODS, codes))
shutil.rmtree(f"{OUT}/audio", ignore_errors=True)
os.makedirs(f"{OUT}/audio"); os.makedirs(PRIVATE, exist_ok=True)
items, key = [], {}
for s in pool:
    ref = by_method["GATAS"][s]
    conditions = {"clean": (ref["gt_wav"], ref["original_transcription"])}
    for m in METHODS:
        r = by_method[m][s]
        conditions[m] = (r["adv_wav"], r["transcription"])
    for cond, (wav, transcription) in conditions.items():
        item_id = f"i{rng.getrandbits(40):010x}"
        normalize(wav, f"{OUT}/audio/{item_id}.wav")
        items.append({"id": item_id, "code": code_of[cond], "sentence": s, "gt_text": ref["gt_text"], "transcription": shorten(transcription)})
        key[item_id] = {"sentence": s, "condition": cond, "source": os.path.relpath(wav)}
json.dump({"items": items, "conditions_per_sentence": len(conditions), "per_condition": 4}, open(f"{OUT}/items.json", "w"), indent=1)
json.dump({"codes": code_of, "items": key}, open(f"{PRIVATE}/key.json", "w"), indent=1)
print(f"{len(items)} items = {len(pool)} sentences x {len(conditions)} conditions")

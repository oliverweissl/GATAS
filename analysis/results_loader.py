"""
Load experiment results of all methods in one normalized format for the analysis notebooks.

Layout written by the experiment scripts: <results_root>/<Method>/<asr>_<timestamp>/sentence_XXX/[run_Y/]
with one summary JSON (run_summary.json, smack_summary.json or pgd_summary.json), ground_truth.wav and best_*.wav.
"""
import os
import json
import warnings
from glob import glob

METHODS = ("GATAS", "Waveform", "SMACK", "SMACK_targeted", "PGD")


def _record(method: str, path: str) -> dict:
    with open(path) as f:
        d = json.load(f)
    folder = os.path.dirname(path)
    adv = sorted(glob(os.path.join(folder, "best_*.wav")))
    text = d.get("text_data", {})
    nat = d.get("naturalness_scores") or {}

    if "efficiency_metrics" in d:    # GATAS / Waveform (RunLogger)
        sentence_id, run_id = d["metadata"]["sentence_id"], d["metadata"].get("run_id") or 0
        elapsed = d["efficiency_metrics"]["elapsed_time_seconds"]
        generations = d["efficiency_metrics"]["generation_count"]
        pop_size = d["algorithm_parameters"]["pop_size"]
        gt_text, transcription = text["ground_truth_text"], text["asr_transcription"]
    elif "efficiency" in d:          # SMACK (attack_summary)
        sentence_id, run_id = d["metadata"]["sentence_id"], d["metadata"].get("run_id") or 0
        elapsed = d["efficiency"]["elapsed_time_seconds"]
        generations = d["efficiency"]["num_generations"]
        pop_size = d["efficiency"]["pop_size"]
        gt_text, transcription = text["ground_truth_text"], text["asr_transcription"]
    else:                            # PGD (result_writer)
        sentence_id, run_id = d["sentence_id"], 0
        elapsed = d["elapsed_seconds"]
        generations = d["params"]["num_generations"]
        pop_size = d["params"]["pop_size"]
        gt_text, transcription = d["gt_text"], d["transcription"]
        text = {"gt_transcription": d.get("gt_transcription"), "target_text": None}

    return {
        "method": method,
        "sentence_id": sentence_id,
        "run_id": run_id,
        "summary": d,
        "summary_path": path,
        "gt_wav": os.path.join(folder, "ground_truth.wav"),
        "adv_wav": adv[0] if adv else None,
        "success": bool(d["success_metrics"]["success"]),
        "metric_scores": d["success_metrics"].get("metric_scores"),
        "gt_text": gt_text,
        "gt_transcription": text.get("gt_transcription"),
        "transcription": transcription,
        "target_text": text.get("target_text"),
        "elapsed_seconds": elapsed,
        "generations": generations,
        "budget": generations * pop_size,
        "utmos": nat.get("utmos_best"),
        "utmos_gt": nat.get("utmos_gt"),
    }


def _experiment_dir(results_root: str, method: str, asr_model: str):
    experiments = sorted(glob(os.path.join(results_root, method, f"{asr_model}_*")))
    if not experiments:
        warnings.warn(f"No {method} results for ASR model '{asr_model}' in {results_root}")
        return None
    if len(experiments) > 1:
        warnings.warn(f"{len(experiments)} {method} experiments for '{asr_model}', using the latest: {experiments[-1]}")
    return experiments[-1]


def load_results(results_root: str = "../outputs/results", asr_model: str = "whisper",
                 methods=METHODS, common_sentences: bool = True) -> dict:
    """
    Returns {method: [record, ...]} sorted by sentence_id, one record per sentence (run 0 if repeated).
    Uses the latest experiment per method for asr_model. With common_sentences, only sentences present
    for every method are kept, so lists are aligned by sentence for paired statistics.
    """
    records = {}
    for method in methods:
        experiment = _experiment_dir(results_root, method, asr_model)
        by_sentence = {}
        if experiment is not None:
            for path in glob(os.path.join(experiment, "**", "*summary.json"), recursive=True):
                record = _record(method, path)
                current = by_sentence.get(record["sentence_id"])
                if current is None or record["run_id"] < current["run_id"]:
                    by_sentence[record["sentence_id"]] = record
        records[method] = by_sentence

    if common_sentences and records:
        common = set.intersection(*(set(r) for r in records.values()))
        for method, by_sentence in records.items():
            dropped = sorted(set(by_sentence) - common)
            if dropped:
                warnings.warn(f"{method}: dropping sentences missing for other methods: {dropped}")
        records = {m: {s: r for s, r in by_sentence.items() if s in common} for m, by_sentence in records.items()}

    return {m: [by_sentence[s] for s in sorted(by_sentence)] for m, by_sentence in records.items()}

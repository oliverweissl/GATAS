"""
Load evaluated results (evaluation.json from scripts/evaluate_results.py) for the analysis notebooks.

Layout: <results_root>/<Method>/<dataset>/<asr>_<timestamp>/sentence_XXX/[run_Y/]evaluation.json
"""
import os
import json
import warnings
from glob import glob

METHODS = ("GATAS", "Waveform", "SMACK", "ALIF", "PGD")          # main comparison (untargeted)
ALL_METHODS = ("GATAS", "GATAS_targeted", "Waveform", "SMACK", "SMACK_targeted", "ALIF", "PGD")
BUDGET_ABLATION = ("GATAS", "GATAS_budget300", "GATAS_budget10000")  # 20x10, 20x15, 100x100
DATASETS = ("harvard", "librispeech", "commands")
ASR_MODELS = ("whisper-tiny", "whisper-large-v3-turbo", "wav2vec2-large")

# Objective names in archives / Pareto fronts -> metric names of evaluation.json
_METRIC_ALIASES = {"WER_TARGET": "TARGET_WER"}


def _record(path: str) -> dict:
    with open(path) as f:
        e = json.load(f)
    scores = e["scores"]
    return {
        **e,
        # flat aliases used by the notebooks
        "gt_wav": e["audio"]["ground_truth"],
        "adv_wav": e["audio"]["adversarial"],
        "pesq": scores["PESQ"],
        "set_overlap": scores["SET_OVERLAP"],
        "wer": scores["WER"],
        "sbert": scores["SBERT_SIMILARITY"],
        "utmos": scores["UTMOS"],
        "utmos_gt": scores["UTMOS_GT"],
        "valid": e["validity"]["valid"],
        "reference_wer": e["validity"]["reference_wer"],
        "budget": e["queries"],
    }


def _experiment_dir(results_root: str, method: str, dataset: str, asr_model: str):
    experiments = sorted(glob(os.path.join(results_root, method, dataset, f"{asr_model}_*")))
    if not experiments:
        warnings.warn(f"No {method} results for {dataset} / {asr_model} in {results_root}")
        return None
    if len(experiments) > 1:
        warnings.warn(f"{len(experiments)} {method} experiments for {dataset} / {asr_model}, using the latest: {experiments[-1]}")
    return experiments[-1]


def load_results(results_root: str = "../outputs/results", dataset: str = "harvard", asr_model: str = "whisper-tiny",
                 methods=METHODS, common_sentences: bool = True) -> dict:
    """
    Returns {method: [record, ...]} sorted by sentence_id, one record per sentence (run 0 if repeated), from the
    latest experiment per method for (dataset, asr_model). With common_sentences, only sentences present for every
    method are kept, so the lists are aligned by sentence for paired statistics.
    """
    records = {}
    for method in methods:
        experiment = _experiment_dir(results_root, method, dataset, asr_model)
        by_sentence = {}
        if experiment is not None:
            paths = glob(os.path.join(experiment, "**", "evaluation.json"), recursive=True)
            if not paths:
                warnings.warn(f"{experiment} has no evaluation.json; run scripts/evaluate_results.py")
            for path in paths:
                record = _record(path)
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


def success_at_thresholds(record: dict, thresholds: dict) -> bool:
    """
    Success of one result under other thresholds ({metric: max value}, metrics: PESQ, SET_OVERLAP, WER, ...).
    GATAS/Waveform succeed if any member of their final Pareto front meets all thresholds (the front is what the
    search returns); single-output baselines are judged on their one adversarial example. WER thresholds are
    lower bounds on divergence and given as {"WER": ("min", value)}.
    """
    def meets(scores):
        for name, threshold in thresholds.items():
            if isinstance(threshold, tuple):
                kind, value = threshold
                if scores.get(name) is None or (kind == "min" and scores[name] < value):
                    return False
            elif scores.get(name) is None or scores[name] > threshold:
                return False
        return True

    front = [{_METRIC_ALIASES.get(k, k): v for k, v in member.items()} for member in record.get("pareto_front") or []]
    candidates = front if front else [record["scores"]]
    return any(meets(c) for c in candidates)


def success_at_budget(record: dict, queries: int, thresholds: dict = None) -> bool:
    """
    Success within a query budget. GATAS/Waveform: some archive member meets the thresholds by generation
    queries // pop_size (archive_history.json). Baselines: success if they used at most `queries`.
    """
    thresholds = thresholds or record["success_thresholds"]
    if record.get("archive_history"):
        with open(record["archive_history"]) as f:
            history = json.load(f)
        generation = min(queries // record["pop_size"], len(history["archive_per_generation"]))
        if generation < 1:
            return False
        names = [_METRIC_ALIASES.get(n, n) for n in history["objectives"]]
        archive = history["archive_per_generation"][generation - 1]
        return any(all(dict(zip(names, member)).get(k, float("inf")) <= v for k, v in thresholds.items()) for member in archive)
    return record["success"] and record["queries"] <= queries


def summary_table(records: dict):
    """Per-method means of the evaluation metrics as a pandas DataFrame."""
    import numpy as np
    import pandas as pd

    rows = {}
    for method, recs in records.items():
        if not recs:
            continue
        rows[method] = {
            "n": len(recs),
            "success %": 100 * np.mean([r["success"] for r in recs]),
            "valid success %": 100 * np.mean([r["valid_success"] for r in recs]),
            "SET_OVERLAP": np.mean([r["set_overlap"] for r in recs]),
            "WER": np.mean([r["wer"] for r in recs]),
            "SBERT sim.": np.mean([r["sbert"] for r in recs]),
            "PESQ fitness": np.mean([r["pesq"] for r in recs]),
            "UTMOS": np.mean([r["utmos"] for r in recs]),
            "ref. ASR WER": np.mean([r["reference_wer"] for r in recs]),
            "runtime [s]": np.mean([r["elapsed_seconds"] for r in recs]),
            "queries": np.mean([r["queries"] for r in recs]),
        }
    return pd.DataFrame(rows).T

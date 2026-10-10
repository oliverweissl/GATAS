"""
Load evaluated results (evaluation.json from scripts/evaluate_results.py) for the analysis notebooks.

Layout: <results_root>/<Method>/<dataset>/<asr>_<timestamp>/sentence_XXX/[run_Y/]evaluation.json
"""
import os
import json
import warnings
from glob import glob

METHODS = ("GATAS", "Waveform", "SMACK", "ALIF", "PGD", "SMACK_targeted")          # main comparison (untargeted)
ALL_METHODS = ("GATAS", "GATAS_targeted", "Waveform", "SMACK", "SMACK_targeted", "ALIF", "PGD")

# Objective names in archives / Pareto fronts -> metric names of evaluation.json
_METRIC_ALIASES = {"WER_TARGET": "TARGET_WER"}


def _record(path: str) -> dict:
    with open(path) as f:
        e = json.load(f)
    scores = e["scores"]
    return {
        **e,
        # flat aliases used by the notebooks
        "gt_wav": e["audio"]["original"],
        "adv_wav": e["audio"]["adversarial"],
        "wer": scores["WER"],  # lexical distance
        "semantic": scores["SEMANTIC_DISTANCE"],
        "spectrogram": scores["SPECTROGRAM_DISTANCE"],
        "utmos": scores["UTMOS"],
        "utmos_original": scores["UTMOS_ORIGINAL"],
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
                 methods=METHODS) -> dict:
    """
    Returns {method: [record, ...]} sorted by sentence_id, one record per sentence (run 0 if repeated), from the
    latest experiment per method for (dataset, asr_model). No sentences are filtered: methods may cover different
    sentences; paired statistics must match records by sentence_id.
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

    return {m: [by_sentence[s] for s in sorted(by_sentence)] for m, by_sentence in records.items()}


def success_at_budget(record: dict, queries: int, thresholds: dict = None) -> bool:
    """
    Success within a query budget. GATAS/Waveform: some archive member meets `thresholds` (required; archives hold
    only the optimizer objectives, not transcriptions) by generation queries // pop_size (archive_history.json).
    Baselines: success if they used at most `queries`.
    """
    if record.get("archive_history"):
        if not thresholds:
            raise ValueError("success_at_budget needs objective thresholds for archive-based methods")
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
    """Per-method success rate and mean distances as a pandas DataFrame."""
    import numpy as np
    import pandas as pd

    rows = {}
    for method, recs in records.items():
        if not recs:
            continue
        rows[method] = {
            "n": len(recs),
            "original fails %": 100 * np.mean([not r["original_passes"] for r in recs]),
            "test case found %": 100 * np.mean([r["success"] for r in recs]),
            "found % (original passes)": 100 * np.mean([r["success"] for r in recs if r["original_passes"]] or [np.nan]),
            "lexical (WER)": np.mean([r["wer"] for r in recs]),
            "semantic": np.mean([r["semantic"] for r in recs]),
            "spectrogram": np.mean([r["spectrogram"] for r in recs]),
            "UTMOS": np.mean([r["utmos"] for r in recs]),
        }
    return pd.DataFrame(rows).T


def pareto_plot(ax, points: dict, x_key: str, y_key: str, x_higher_better: bool, y_higher_better: bool = True):
    """
    Draws one marker per method and the Pareto front (line through the non-dominated methods).
    `points` is {label: {x_key: float, y_key: float}}, e.g. rows of `summary_table(...).to_dict("index")`.
    Returns the labels on the front.
    """
    sx, sy = (1 if x_higher_better else -1), (1 if y_higher_better else -1)
    labels = list(points)
    xs = {l: sx * points[l][x_key] for l in labels}
    ys = {l: sy * points[l][y_key] for l in labels}
    front = [l for l in labels if not any(
        xs[o] >= xs[l] and ys[o] >= ys[l] and (xs[o] > xs[l] or ys[o] > ys[l]) for o in labels if o != l)]
    front.sort(key=lambda l: xs[l])
    for l in labels:
        ax.scatter(points[l][x_key], points[l][y_key], s=90, zorder=3, marker="o" if l in front else "x", label=l)
    ax.plot([points[l][x_key] for l in front], [points[l][y_key] for l in front], color="black", alpha=0.5, zorder=2)
    ax.grid(alpha=0.3)
    return front

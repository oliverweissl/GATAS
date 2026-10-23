"""Load evaluation.json files (scripts/evaluate_results.py) for the analysis notebooks.

Layout: <results_root>/<Method>/<dataset>/<asr>_<timestamp>/sentence_XXX/[run_Y/]evaluation.json
"""
import json
import os
import warnings
from glob import glob

import numpy as np

METHODS = ("GATAS", "Waveform", "SMACK_untargeted", "ALIF", "PGD", "SMACK")
# method -> result folder on the commands dataset (intent task); PGD stays untargeted
INTENT_FOLDERS = {"GATAS": "GATAS_intent", "Waveform": "Waveform_intent", "SMACK": "SMACK", "ALIF": "ALIF_intent", "PGD": "PGD",
                  "GATAS_llm": "GATAS_llm"}
MODELS = ("whisper-tiny", "wav2vec2-large")
MODEL_LABELS = {"whisper-tiny": "Whisper-tiny", "wav2vec2-large": "wav2vec2-large"}


def _record(path: str) -> dict:
    with open(path) as f:
        e = json.load(f)
    scores = e["scores"]
    return {**e, "gt_wav": e["audio"]["original"], "adv_wav": e["audio"]["adversarial"], "wer": scores["WER"],
            "semantic": scores["SEMANTIC_DISTANCE"], "spectrogram": scores["SPECTROGRAM_DISTANCE"], "utmos": scores["UTMOS"],
            "utmos_original": scores["UTMOS_ORIGINAL"], "budget": e["queries"]}


def _experiment_dir(results_root, method, dataset, asr_model):
    experiments = sorted(glob(os.path.join(results_root, method, dataset, f"{asr_model}_*")))
    if not experiments:
        warnings.warn(f"No {method} results for {dataset} / {asr_model} in {results_root}")
        return None
    if len(experiments) > 1:
        warnings.warn(f"{len(experiments)} {method} experiments for {dataset} / {asr_model}, using the latest: {experiments[-1]}")
    return experiments[-1]


def load_results(results_root: str = "../outputs/results", dataset: str = "harvard", asr_model: str = "whisper-tiny",
                 methods=METHODS, exclude_original_fails: bool = False) -> dict:
    """{method: [record, ...]} sorted by sentence_id (run 0 if repeated), from the latest experiment per method.
    Sentences whose clean audio already fails the ASR model are dropped with `exclude_original_fails`."""
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
    return {m: [by_sentence[s] for s in sorted(by_sentence) if by_sentence[s]["original_passes"] or not exclude_original_fails]
            for m, by_sentence in records.items()}


def load_intent_results(results_root: str = "../outputs/results", asr_model: str = "whisper-tiny",
                        exclude_original_wrong: bool = False) -> dict:
    """{method: records} on the commands dataset; `exclude_original_wrong` drops commands whose clean intent is already wrong."""
    folders = load_results(results_root, dataset="commands", asr_model=asr_model, methods=tuple(INTENT_FOLDERS.values()))
    return {method: [r for r in folders[folder] if r["intent"] and (r["intent"]["original_correct"] or not exclude_original_wrong)]
            for method, folder in INTENT_FOLDERS.items()}


def asr_found(r: dict) -> bool:
    return bool(r["success"])


def intent_hit(r: dict) -> bool:
    return bool(r["intent"]["changed"] and r["intent"]["target_match"])


def load_all(results_root: str = "../outputs/results", models=MODELS, exclude_baseline_failures: bool = True):
    """(asr, intent), each {model: {method: records}} without empty methods; sentences whose clean input already fails are dropped."""
    asr, intent = {}, {}
    for model in models:
        a = load_results(results_root, dataset="harvard", asr_model=model)
        i = load_intent_results(results_root, asr_model=model)
        if exclude_baseline_failures:
            a = {m: [r for r in rs if r["original_passes"]] for m, rs in a.items()}
            i = {m: [r for r in rs if r["intent"]["original_correct"]] for m, rs in i.items()}
        asr[model] = {m: rs for m, rs in a.items() if rs}
        intent[model] = {m: rs for m, rs in i.items() if rs}
    return asr, intent


def validity_threshold(records: list, percentile: float = 2.5) -> float:
    """Lower bound of the central 95% UTMOS interval of the original audio (m-valid threshold)."""
    return float(np.percentile([r["utmos_original"] for r in records], percentile))


def task_thresholds(asr: dict, intent: dict, models=MODELS) -> dict:
    """m-valid UTMOS threshold per task; the original audio does not depend on the attacked model."""
    out = {}
    for task, data in (("ASR", asr), ("Intent", intent)):
        model = next(m for m in models if data.get(m, {}).get("GATAS"))
        out[task] = validity_threshold(data[model]["GATAS"])
    return out


PCT = r"\%"
_NAMES = {  # method -> LaTeX name for plots
    "GATAS": r"\textsc{Gatas}", "GATAS_llm": r"\textsc{Gatas}\textsuperscript{$\dagger$}", "SMACK": r"\textsc{Smack}",
    "SMACK_untargeted": r"\textsc{Smack}\textsuperscript{$\ddagger$}", "Waveform": r"\textsc{Waveform}", "ALIF": r"\textsc{Alif}", "PGD": r"\textsc{Pgd}",
}
METHOD_ORDER = ("GATAS", "GATAS_llm", "SMACK", "SMACK_untargeted", "Waveform", "ALIF", "PGD")


def tex_name(method: str) -> str:
    return _NAMES.get(method, method.replace("_", r"\_"))


def setup_latex_plots(font_size: int = 18):
    import matplotlib.pyplot as plt
    plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": font_size})


def ordered(methods):
    return sorted(methods, key=lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else len(METHOD_ORDER))


def method_color(method: str):
    import matplotlib.pyplot as plt
    return plt.get_cmap("tab10")(METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER))

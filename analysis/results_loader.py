"""
Load evaluated results (evaluation.json from scripts/evaluate_results.py) for the analysis notebooks.

Layout: <results_root>/<Method>/<dataset>/<asr>_<timestamp>/sentence_XXX/[run_Y/]evaluation.json
"""
import os
import json
import warnings
from glob import glob

METHODS = ("GATAS", "Waveform", "SMACK_untargeted", "ALIF", "PGD", "SMACK")          # main comparison
# Intent matching task (commands dataset): the same methods as in the ASR task (harvard), run with the objective toward the
# target intent. Their result folders carry an `_intent` suffix (run_*_intent.sh); PGD is white-box on Whisper and stays untargeted.
INTENT_FOLDERS = {"GATAS": "GATAS_intent", "Waveform": "Waveform_intent", "SMACK": "SMACK", "ALIF": "ALIF_intent", "PGD": "PGD",
                  "GATAS_llm": "GATAS_llm"}
ALL_METHODS = ("GATAS", "GATAS_targeted", "Waveform", "SMACK_untargeted", "SMACK", "ALIF", "PGD",
               "GATAS_intent", "Waveform_intent", "ALIF_intent", "GATAS_llm")

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
                 methods=METHODS, exclude_original_fails: bool = False) -> dict:
    """
    Returns {method: [record, ...]} sorted by sentence_id, one record per sentence (run 0 if repeated), from the
    latest experiment per method for (dataset, asr_model). Sentences whose clean audio already fails the ASR model
    (original_passes false) are kept by default; set `exclude_original_fails` to drop them (report their share separately).
    Methods may still cover different sentences; paired statistics must match records by sentence_id.
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

    return {m: [by_sentence[s] for s in sorted(by_sentence) if by_sentence[s]["original_passes"] or not exclude_original_fails]
            for m, by_sentence in records.items()}


def load_intent_results(results_root: str = "../outputs/results", asr_model: str = "whisper-tiny",
                        exclude_original_wrong: bool = False) -> dict:
    """
    Intent matching task: {method: records} on the commands dataset, methods under their plain names (GATAS, ...).
    Sentences whose clean transcription already maps to a different intent than the reference are kept unless
    `exclude_original_wrong` (report their share separately).
    """
    folders = load_results(results_root, dataset="commands", asr_model=asr_model, methods=tuple(INTENT_FOLDERS.values()),
                           exclude_original_fails=False)
    return {method: [r for r in folders[folder] if r["intent"] and (r["intent"]["original_correct"] or not exclude_original_wrong)]
            for method, folder in INTENT_FOLDERS.items()}


MODELS = ("whisper-tiny", "wav2vec2-large")  # attacked ASR models of the paper; the first one is the primary model
MODEL_LABELS = {"whisper-tiny": "Whisper-tiny", "wav2vec2-large": "wav2vec2-large"}


def asr_found(r: dict) -> bool:
    """ASR task: a test case is found. Single place to change if the success criterion changes."""
    return bool(r["success"])


def intent_hit(r: dict) -> bool:
    """Intent task: the intent changed to the target intent."""
    return bool(r["intent"]["changed"] and r["intent"]["target_match"])


def load_all(results_root: str = "../outputs/results", models=MODELS, exclude_baseline_failures: bool = True):
    """
    Both tasks for every attacked model: returns (asr, intent), each {model: {method: [record, ...]}} with empty methods
    dropped. With `exclude_baseline_failures`, sentences whose clean input already fails (ASR: transcription != ground
    truth; intent: clean intent != reference) are dropped per model.
    """
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


def task_thresholds(asr: dict, intent: dict, models=MODELS) -> dict:
    """m-valid UTMOS threshold per task. The ground-truth audio does not depend on the attacked model, so the threshold is
    computed once on the first model that has GATAS results."""
    out = {}
    for task, data in (("ASR", asr), ("Intent", intent)):
        model = next(m for m in models if data.get(m, {}).get("GATAS"))
        out[task] = validity_threshold(data[model]["GATAS"])
    return out


def validity_threshold(records: list, percentile: float = 2.5) -> float:
    """
    UTMOS threshold of a valid test case: the lower bound of the central (100 - 2 * percentile) % interval of the UTMOS
    of the original (ground-truth) audio of the task. `records` are the records of one method (all methods share the
    original audios), restricted to the sentences used in the analysis.
    """
    import numpy as np
    return float(np.percentile([r["utmos_original"] for r in records], percentile))


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


# ---------------------------------------------------------------- display names / plot style
PCT = r"\%"
_DAGGER = r"\textsuperscript{$\dagger$}"
_DDAGGER = r"\textsuperscript{$\ddagger$}"
# folder/key name -> (LaTeX name for plots, plain name for tables). SMACK is the original (native targeted) SMACK, SMACK_untargeted our untargeted adaptation for the ASR task.
_NAMES = {
    "GATAS": (r"\textsc{Gatas}", "GATAS"),
    "GATAS_llm": (r"\textsc{Gatas}" + _DAGGER, "GATAS†"),
    "SMACK": (r"\textsc{Smack}", "SMACK"),
    "SMACK_untargeted": (r"\textsc{Smack}" + _DDAGGER, "SMACK‡"),
    "Waveform": (r"\textsc{Waveform}", "Waveform"),
    "ALIF": (r"\textsc{Alif}", "ALIF"),
    "PGD": (r"\textsc{Pgd}", "PGD"),
}


def tex_name(method: str) -> str:
    return _NAMES.get(method, (method.replace("_", r"\_"),))[0]


def setup_latex_plots(font_size: int = 18):
    import matplotlib.pyplot as plt
    plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": font_size})


# plotting order: related methods next to each other (GATAS variants, SMACK variants, then the rest)
METHOD_ORDER = ("GATAS", "GATAS_llm", "SMACK", "SMACK_untargeted", "Waveform", "ALIF", "PGD")


def ordered(methods):
    return sorted(methods, key=lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else len(METHOD_ORDER))


def method_color(method: str):
    """One fixed colour per method, identical across tasks and figures."""
    import matplotlib.pyplot as plt
    i = METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER)
    return plt.get_cmap("tab10")(i)

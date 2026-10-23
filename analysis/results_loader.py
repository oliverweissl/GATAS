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


def summary_table(records: dict):
    """Per-method success rate and mean distances as a pandas DataFrame."""
    import numpy as np
    import pandas as pd

    rows = {}
    for method, recs in records.items():
        if not recs:
            continue
        rows[method] = {
            "test case found %": 100 * np.mean([r["success"] for r in recs]),
            "lexical (WER)": np.mean([r["wer"] for r in recs]),
            "semantic": np.mean([r["semantic"] for r in recs]),
            "spectrogram": np.mean([r["spectrogram"] for r in recs]),
            "UTMOS": np.mean([r["utmos"] for r in recs]),
        }
    return pd.DataFrame(rows).T


def intent_summary_table(records: dict):
    """
    Intent matching per method (records from load_intent_results).
      intent changed %: share of sentences where the adversarial transcription is classified differently than the clean one
      target intent %:  share of the changed intents that are the target intent (untargeted methods: by chance)
    """
    import numpy as np
    import pandas as pd

    rows = {}
    for method, recs in records.items():
        recs = [r for r in recs if r.get("intent")]
        if not recs:
            continue
        changed = [r for r in recs if r["intent"]["changed"]]
        rows[method] = {
            "intent changed %": 100 * len(changed) / len(recs),
            "target intent %": 100 * np.mean([r["intent"]["target_match"] for r in changed]) if changed else np.nan,
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


def first_success_queries(record: dict, thresholds_from_summary: bool = True) -> float:
    """
    Queries until the first test case (inf if none). Archive methods (GATAS/Waveform): first generation whose archive has a
    member meeting the run's objective thresholds (run_summary.json) times pop_size, as a proxy since archives store optimizer
    objectives only; if the final candidate is a success but no archive member meets them, the full budget. Baselines (no
    per-query history): total queries used if successful (upper bound).
    """
    if not record["success"]:
        return float("inf")
    if not record.get("archive_history"):
        return float(record["queries"])
    with open(record["summary_path"]) as f:
        thresholds = json.load(f)["objectives"]["thresholds"]
    with open(record["archive_history"]) as f:
        history = json.load(f)
    names = [_METRIC_ALIASES.get(n, n) for n in history["objectives"]]
    for g, archive in enumerate(history["archive_per_generation"], 1):
        if any(all(dict(zip(names, m)).get(k, float("inf")) <= v for k, v in thresholds.items()) for m in archive):
            return float(g * record["pop_size"])
    return float(record["queries"])


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


def plain_name(method: str) -> str:
    return _NAMES.get(method, (None, method))[1]


def pretty(df):
    """Plain display names for method rows / the `vs` column of a table."""
    df = df.rename(index=plain_name)
    if "vs" in df:
        df = df.assign(vs=df["vs"].map(plain_name))
    return df


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

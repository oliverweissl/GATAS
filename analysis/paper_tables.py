"""
LaTeX tables of the paper with both attacked ASR models side by side (column groups per model).
Used by RQ1_Effectiveness.ipynb and RQ3_Efficiency.ipynb. A missing method / model is shown as `--`.
"""
import numpy as np

from results_loader import MODELS, MODEL_LABELS
from stats_utils import wilson_ci

TEX_NAME = {"GATAS": r"\tool", "GATAS_llm": r"\gllm", "Waveform": r"\wv", "SMACK_untargeted": r"\smack$^\ddagger$", "SMACK": r"\smack",
            "ALIF": r"\alif", "PGD": r"\pgd"}
ASR_ORDER = ("GATAS", "Waveform", "SMACK_untargeted", "SMACK", "ALIF", "PGD")
INTENT_ORDER = ("GATAS", "Waveform", "SMACK", "ALIF", "PGD", "GATAS_llm")
NOTE_PGD = r"\pgd is white-box on Whisper and has no wav2vec2 run."


def _num(x, digits=1):
    x = max(x, 0.0)  # no "-0.0" from a rounded interval bound
    return "100" if round(x, digits) == 100 else f"{x:.{digits}f}"


def rate_col(name, success, valid=False, ci=True, higher=True):
    """Share [%] of sentences with a success (and m-valid if `valid`); higher is better."""
    def f(recs, thr):
        n = len(recs)
        k = sum(success(r) and (not valid or r["utmos"] >= thr) for r in recs)
        suffix = ""
        if ci:
            lo, hi = wilson_ci(k, n)
            suffix = f" [{_num(100 * lo)}, {_num(100 * hi)}]"
        return 100 * k / n, _num(100 * k / n), suffix
    return name, f, higher


def mean_col(name, key, digits, higher):
    get = key if callable(key) else (lambda r: r[key])

    def f(recs, thr):
        v = float(np.mean([get(r) for r in recs]))
        return v, f"{v:.{digits}f}", ""
    return name, f, higher


SEMANTIC_CACHE, JUDGE = "../outputs/semantic_scores.json", "gemma4:31b"
_sem_cache = {}


def verdict(r):
    """Three-way LLM verdict of an ASR record: preserved / changed / garbled (scripts/semantic_rescore.py --llm3)."""
    import json
    if not _sem_cache:
        _sem_cache.update(json.load(open(SEMANTIC_CACHE)))
    return _sem_cache[json.dumps([r["gt_text"], r["transcription"] or ""])][f"llm3:{JUDGE}"]


def _top(columns, models, data, methods, thr):
    """Cell values {(model, col, method): (val, num, suffix)} and the best value per (model, col)."""
    cells, best = {}, {}
    for model in models:
        for ci, (_, fn, higher) in enumerate(columns):
            vals = []
            for m in methods:
                recs = data.get(model, {}).get(m)
                if recs:
                    cells[model, ci, m] = fn(recs, thr)
                    vals.append(cells[model, ci, m][0])
            if higher is not None and len(set(vals)) > 1:  # no bold when all methods tie or the column has no direction
                best[model, ci] = max(vals) if higher else min(vals)
    return cells, best


def effectiveness_tex(data, methods, columns, thr, caption, label, models=MODELS, names=TEX_NAME, footnote=None):
    """Effectiveness table: one column group per model. `data` is {model: {method: records}}."""
    cells, best = _top(columns, models, data, methods, thr)
    k = len(columns)
    head1 = " & " + " & ".join(rf"\multicolumn{{{k}}}{{c}}{{{MODEL_LABELS[m]}}}" for m in models) + r" \\"
    rules = "".join(rf"\cmidrule(lr){{{2 + g * k}-{1 + (g + 1) * k}}}" for g in range(len(models)))
    head2 = " & " + " & ".join(c[0] for _ in models for c in columns) + r" \\ \midrule"
    rows = []
    for m in methods:
        if not any(data.get(model, {}).get(m) for model in models):
            continue
        out = []
        for model in models:
            for ci, (_, _, _) in enumerate(columns):
                c = cells.get((model, ci, m))
                if c is None:
                    out.append("--")
                    continue
                num = rf"\textbf{{{c[1]}}}" if (model, ci) in best and abs(c[0] - best[model, ci]) < 1e-9 else c[1]
                out.append(num + c[2])
        rows.append(f"{names[m]} & " + " & ".join(out) + r" \\")
    star = "*" if k * len(models) >= 8 else ""
    cols = "l" + "c" * (k * len(models))
    lines = [rf"\begin{{table{star}}}[t]", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\centering",
             rf"\resizebox{{\{'textwidth' if star else 'columnwidth'}}}{{!}}{{%", rf"\begin{{tabular}}{{{cols}}}\toprule", head1, rules, head2,
             *rows, r"\bottomrule", r"\end{tabular}}", rf"\end{{table{star}}}"]
    return "\n".join(lines)


ASR_COLUMNS_MAIN = None  # built lazily to avoid importing asr_found at import time


def asr_columns(full=False):
    """Main: share of sentences with a changed / garbled transcription (LLM verdict), m-valid changed, UTMOS.
    Full adds the WER > 0 criterion and the distances."""
    from results_loader import asr_found
    changed = lambda r: verdict(r) == "changed"
    cols = [rate_col(r"Changed [\%] $\uparrow$", changed), rate_col(r"Garbled [\%]", lambda r: verdict(r) == "garbled", ci=False, higher=None),
            rate_col(r"M-valid changed [\%] $\uparrow$", changed, valid=True, ci=False)]
    if full:
        cols = [rate_col(r"WER$>0$ [\%]", asr_found, ci=False)] + cols + [mean_col(r"WER $\downarrow$", "wer", 3, False),
                mean_col(r"Semantic $\downarrow$", "semantic", 3, False), mean_col(r"$d_F$-Spec $\downarrow$", "spectrogram", 3, False)]
    return cols + [mean_col(r"UTMOS $\uparrow$", "utmos", 2, True)]


def intent_columns(full=False):
    from results_loader import intent_hit
    cols = [rate_col(r"Changed [\%] $\uparrow$", lambda r: r["intent"]["changed"]),
            rate_col(r"Target hit [\%] $\uparrow$", intent_hit, ci=full),
            rate_col(r"M-valid hit [\%] $\uparrow$", intent_hit, valid=True, ci=False)]
    if full:
        cols += [mean_col(r"WER $\downarrow$", "wer", 3, False), mean_col(r"Semantic $\downarrow$", "semantic", 3, False),
                 mean_col(r"$d_F$-Spec $\downarrow$", "spectrogram", 3, False)]
    return cols + [mean_col(r"UTMOS $\uparrow$", "utmos", 2, True)]


def stats_tex(rows, tasks, models=MODELS, names=TEX_NAME):
    """
    Paired-comparison table. rows[model][task] = {method: (p_found, d_found, p_utmos, d_utmos)} as LaTeX strings;
    `tasks` = [(task key, label)], the first method of each task's reference is stated in the label.
    """
    cols = "ll" + "rrrr" * len(models)
    head1 = " & & " + " & ".join(rf"\multicolumn{{4}}{{c}}{{{MODEL_LABELS[m]}}}" for m in models) + r" \\"
    rules = "".join(rf"\cmidrule(lr){{{3 + g * 4}-{6 + g * 4}}}" for g in range(len(models)))
    head2 = " & & " + " & ".join(r"\multicolumn{2}{c}{Found / hit} & \multicolumn{2}{c}{UTMOS}" for _ in models) + r" \\"
    rules2 = "".join(rf"\cmidrule(lr){{{3 + g * 4}-{4 + g * 4}}}\cmidrule(lr){{{5 + g * 4}-{6 + g * 4}}}" for g in range(len(models)))
    head3 = "Task & vs. & " + " & ".join(r"$p$ & $d$ & $p$ & $d$" for _ in models) + r" \\ \midrule"
    body = []
    for ti, (task, label) in enumerate(tasks):
        methods = [m for m in dict.fromkeys(m for model in models for m in rows[model].get(task, {}))]
        for k, m in enumerate(methods):
            lead = rf"\multirow{{{len(methods)}}}{{*}}{{{label}}}" if k == 0 else ""
            out = []
            for model in models:
                v = rows[model].get(task, {}).get(m)
                out += list(v) if v else ["--"] * 4
            body.append(f"{lead} & {names[m]} & " + " & ".join(out) + r" \\" + (r" \midrule" if k == len(methods) - 1 and ti < len(tasks) - 1 else ""))
    lines = [r"\begin{table*}[t]", r"\caption{Paired comparison of the reference method with each other method (RQ\textsubscript{1}).}",
             r"\label{tab:stats}", r"\centering", r"\resizebox{\textwidth}{!}{%", rf"\begin{{tabular}}{{{cols}}}\toprule",
             head1, rules, head2, rules2, head3, *body, r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    return "\n".join(lines)


def efficiency_tex(tasks, thr, models=MODELS, names=TEX_NAME):
    """
    Efficiency table. tasks = {task key ("ASR"/"Intent"): {model: {method: records}}} with `runtime` set on every record.
    Per task and model: time [s] (mean) and queries per m-valid success (mean queries / m-valid success rate).
    """
    from results_loader import asr_found, intent_hit
    success = {"ASR": asr_found, "Intent": intent_hit}
    cell, best = {}, {}
    for task, data in tasks.items():
        for model in models:
            for m, recs in data.get(model, {}).items():
                n = len(recs)
                k = sum(success[task](r) and r["utmos"] >= thr[task] for r in recs)
                cell[task, model, m] = (float(np.mean([r["runtime"] for r in recs])),
                                        float(np.mean([r["queries"] for r in recs])) / (k / n) if k else float("inf"))
            vals = [cell[t, mo, m] for (t, mo, m) in cell if t == task and mo == model]
            if vals:
                best[task, model] = (min(v[0] for v in vals), min(v[1] for v in vals))

    def fmt(task, model, m):
        c = cell.get((task, model, m))
        if c is None:
            return ["--", "--"]
        t = f"{c[0]:.1f}"
        q = r"$\infty$" if np.isinf(c[1]) else f"{round(c[1]):,}".replace(",", r"\,")
        if abs(c[0] - best[task, model][0]) < 1e-9:
            t = rf"\textbf{{{t}}}"
        if np.isfinite(c[1]) and abs(c[1] - best[task, model][1]) < 1e-9:
            q = rf"\textbf{{{q}}}"
        return [t, q]

    task_labels = {"ASR": "ASR", "Intent": "Intent (target hit)"}
    g = len(models)
    head1 = " & " + " & ".join(rf"\multicolumn{{{2 * g}}}{{c}}{{{task_labels[t]}}}" for t in tasks) + r" \\"
    rules1 = "".join(rf"\cmidrule(lr){{{2 + i * 2 * g}-{1 + (i + 1) * 2 * g}}}" for i in range(len(tasks)))
    head2 = " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{MODEL_LABELS[mo]}}}" for _ in tasks for mo in models) + r" \\"
    rules2 = "".join(rf"\cmidrule(lr){{{2 + i * 2}-{3 + i * 2}}}" for i in range(len(tasks) * g))
    head3 = " & " + " & ".join(r"Time [s] & Q/m-valid" for _ in range(len(tasks) * g)) + r" \\ \midrule"
    order = [m for m in dict.fromkeys(ASR_ORDER[:1] + INTENT_ORDER[-1:] + ASR_ORDER[1:])]
    rows = []
    for m in order:
        if not any((t, mo, m) in cell for t in tasks for mo in models):
            continue
        out = [x for t in tasks for mo in models for x in fmt(t, mo, m)]
        rows.append(f"{names[m]} & " + " & ".join(out) + r" \\")
    cols = "l" + "cc" * (len(tasks) * g)
    lines = [r"\begin{table*}[t]", r"\caption{Efficiency (RQ\textsubscript{3}). Mean runtime per sentence and queries per m-valid success; "
             r"$\infty$: no m-valid success. All methods use a budget of 300 queries (\smack stops earlier). "
             r"Runtimes are comparable between methods, not between models.}",
             r"\label{tab:efficiency}", r"\centering", r"\resizebox{\textwidth}{!}{%", rf"\begin{{tabular}}{{{cols}}}\toprule",
             head1, rules1, head2, rules2, head3, *rows, r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    return "\n".join(lines)

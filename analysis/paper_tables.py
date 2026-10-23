"""
LaTeX tables of the paper with both attacked ASR models side by side (column groups per model).
Used by RQ1_Effectiveness.ipynb and RQ3_Efficiency.ipynb. A missing method / model is shown as `--`.
"""
import numpy as np

from results_loader import MODELS, MODEL_LABELS
from stats_utils import wilson_ci, paired_values, paired_compare, holm

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

    reference = {"ASR": "GATAS", "Intent": "GATAS_llm"}
    faster = {(t, mo): significance(tasks[t].get(mo, {}), reference[t], lambda r: r["runtime"], False, False) for t in tasks for mo in models}

    def fmt(task, model, m):
        c = cell.get((task, model, m))
        if c is None:
            return ["--", "--"]
        t = f"{c[0]:.1f}"
        q = r"$\infty$" if np.isinf(c[1]) else f"{round(c[1]):,}".replace(",", r"\,")
        if abs(c[0] - best[task, model][0]) < 1e-9:
            t = rf"\textbf{{{t}}}"
        sig = faster[task, model].get(m)
        if sig and m not in OWN and sig[0] < ALPHA and sig[1] > 0:
            t += bullet(sig[1])
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
             r"Runtimes are comparable between methods, not between models. "
             r"The reference method is \tool on ASR and \gllm on intent; bullets mark where it is significantly faster. " + LEGEND + "}",
             r"\label{tab:efficiency}", r"\centering", r"\resizebox{\textwidth}{!}{%", rf"\begin{{tabular}}{{{cols}}}\toprule",
             head1, rules1, head2, rules2, head3, *rows, r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------------------------------------------
# RQ1: one table, both tasks (panels), column groups per SUT. Colored bullet: the reference method (GATAS on ASR, GATAS_llm on intent) is
# significantly better than this baseline in this SUT / task / measure (one-sided paired test, Holm-corrected over the baselines,
# p < 0.05); the bullet color gives the size of the paired Cohen's d_z (positive: reference better).
# ----------------------------------------------------------------------------------------------------------------------------------
ALPHA = 0.05
# effect size of a significant difference as a colored bullet (Cohen's d_z: small <= 0.5 < medium <= 1 < large)
BULLET_COLORS = ((1.0, "green!50!black"), (0.5, "blue!70"), (0.0, "orange!80!black"))
LEGEND = (r"A colored bullet marks a significant difference to the reference method (one-sided paired test, Holm-corrected, $p<0.05$) and "
          r"the size of the paired Cohen's $d_z$: \textcolor{orange!80!black}{$\bullet$}~small ($d_z\le0.5$), \textcolor{blue!70}{$\bullet$}~medium "
          r"($0.5<d_z\le1$), \textcolor{green!50!black}{$\bullet$}~large ($d_z>1$).")


def bullet(d):
    color = next(c for lo, c in BULLET_COLORS if d > lo or lo == 0.0)
    return rf"\,\textcolor{{{color}}}{{$\bullet$}}"


def _rate(f):
    return lambda r: float(f(r))


def rq1_measures(task, thr):
    """[(header, getter, higher is better, binary, digits)] of the RQ1 table for a task; thr = m-valid UTMOS threshold."""
    from results_loader import asr_found, intent_hit
    dist = [(r"WER $\downarrow$", lambda r: r["wer"], False, False, 3), (r"Semantic $\downarrow$", lambda r: r["semantic"], False, False, 3),
            (r"$d_F$-Spec $\downarrow$", lambda r: r["spectrogram"], False, False, 3), (r"UTMOS $\uparrow$", lambda r: r["utmos"], True, False, 2)]
    if task == "ASR":
        return [(r"Found [\%] $\uparrow$", _rate(asr_found), True, True, 1)] + dist + [
            (r"M-valid [\%] $\uparrow$", _rate(lambda r: asr_found(r) and r["utmos"] >= thr), True, True, 1)]
    return [(r"Changed [\%] $\uparrow$", _rate(lambda r: r["intent"]["changed"]), True, True, 1),
            (r"Target hit [\%] $\uparrow$", _rate(intent_hit), True, True, 1)] + dist + [
        (r"M-valid hit [\%] $\uparrow$", _rate(lambda r: intent_hit(r) and r["utmos"] >= thr), True, True, 1)]


def significance(records, reference, getter, higher, binary):
    """{method: (Holm-adjusted one-sided p, d_z)} of the reference against every other method; d_z > 0: reference better."""
    others = [m for m in records if m != reference and records[m]]
    if not records.get(reference):
        return {}
    ps, ds = [], []
    for m in others:
        x, y = paired_values(records, reference, m, getter)
        if len(x) < 2:
            ps.append(1.0); ds.append(float("nan")); continue
        ps.append(paired_compare(x, y, binary=binary, alternative="greater" if higher else "less")["p"])
        diff = (x - y) if higher else (y - x)
        sd = diff.std(ddof=1)
        ds.append(float(diff.mean() / sd) if sd > 0 else float("nan"))
    return {m: (p, d) for m, p, d in zip(others, holm(ps), ds)}


OWN = ("GATAS", "GATAS_llm")  # variants of the proposed method are not baselines and never get a bullet


def _panel(data, methods, measures, reference, models, names):
    k = len(measures)
    stat = {(mo, ci): significance(data.get(mo, {}), reference, getter, higher, binary)
            for mo in models for ci, (_, getter, higher, binary, _) in enumerate(measures)}
    value = lambda recs, getter, binary: float(np.mean([getter(r) for r in recs])) * (100 if binary else 1)
    best = {}
    for mo in models:
        for ci, (_, getter, higher, binary, _) in enumerate(measures):
            vals = [value(data[mo][m], getter, binary) for m in methods if data.get(mo, {}).get(m)]
            if len(set(vals)) > 1:
                best[mo, ci] = max(vals) if higher else min(vals)
    rows = []
    for m in methods:
        if not any(data.get(mo, {}).get(m) for mo in models):
            continue
        out = []
        for mo in models:
            recs = data.get(mo, {}).get(m)
            for ci, (_, getter, higher, binary, digits) in enumerate(measures):
                if not recs:
                    out.append("--"); continue
                v = value(recs, getter, binary)
                text = "100" if binary and round(v, digits) == 100 else f"{v:.{digits}f}"
                if (mo, ci) in best and abs(v - best[mo, ci]) < 1e-9:
                    text = rf"\textbf{{{text}}}"
                sig = stat[mo, ci].get(m)
                if sig and m not in OWN and sig[0] < ALPHA and sig[1] > 0:
                    text += bullet(sig[1])
                out.append(text)
        rows.append(f"{names[m]} & " + " & ".join(out) + r" \\")
    head1 = " & " + " & ".join(rf"\multicolumn{{{k}}}{{c}}{{{MODEL_LABELS[m]}}}" for m in models) + r" \\"
    rules = "".join(rf"\cmidrule(lr){{{2 + g * k}-{1 + (g + 1) * k}}}" for g in range(len(models)))
    head2 = " & " + " & ".join(c[0] for _ in models for c in measures) + r" \\ \midrule"
    return "\n".join([rf"\begin{{tabular}}{{l{'c' * (k * len(models))}}}\toprule", head1, rules, head2, *rows, r"\bottomrule", r"\end{tabular}"])


def rq1_tex(asr, intent, thr, models=MODELS, asr_names=TEX_NAME, intent_names=None):
    """The RQ1 table: panel (a) ASR testing, panel (b) intent matching. `asr` / `intent` are {model: {method: records}}."""
    intent_names = intent_names or {**TEX_NAME, "GATAS_llm": r"\gllm"}
    pa = _panel(asr, ASR_ORDER, rq1_measures("ASR", thr["ASR"]), "GATAS", models, asr_names)
    pi = _panel(intent, ("GATAS", "GATAS_llm") + INTENT_ORDER[1:-1], rq1_measures("Intent", thr["Intent"]), "GATAS_llm", models, intent_names)
    cap = (r"Effectiveness (RQ\textsubscript{1}) on (a) ASR testing and (b) intent matching: test cases found, lexical (WER), semantic and "
           r"spectrogram distance to the clean audio, naturalness (UTMOS) and the share of m-valid test cases. Bold: best value. "
           r"The reference method is \tool on ASR and \gllm on intent matching; " + LEGEND + " " + NOTE_PGD)
    return "\n".join([r"\begin{table*}[t]", rf"\caption{{{cap}}}", r"\label{tab:rq1}", r"\centering", r"\small",
                       r"\textbf{(a) ASR testing}\\[2pt]", r"\resizebox{\textwidth}{!}{%", pa + "}", r"\vspace{6pt}",
                       r"\textbf{(b) Intent matching}\\[2pt]", r"\resizebox{\textwidth}{!}{%", pi + "}", r"\end{table*}"])

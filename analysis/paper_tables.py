"""LaTeX tables of the paper (column groups per attacked model) and the frames of values they are formatted from."""
import json

import numpy as np
import pandas as pd

from results_loader import MODELS, MODEL_LABELS
from stats_utils import wilson_ci, paired_values, paired_compare, holm

TEX_NAME = {"GATAS": r"\tool", "GATAS_llm": r"\gllm", "Waveform": r"\wv", "SMACK_untargeted": r"\smack$^\ddagger$", "SMACK": r"\smack",
            "ALIF": r"\alif", "PGD": r"\pgd"}
ASR_ORDER = ("GATAS", "Waveform", "SMACK_untargeted", "SMACK", "ALIF", "PGD")
INTENT_ORDER = ("GATAS", "Waveform", "SMACK", "ALIF", "PGD", "GATAS_llm")
NOTE_PGD = r"\pgd is white-box on Whisper and has no wav2vec2 run."
SEMANTIC_CACHE, JUDGE = "../outputs/semantic_scores.json", "gemma4:31b"
_sem_cache = {}


def verdict(r):
    """LLM verdict of an ASR record (preserved / changed / garbled), scripts/semantic_rescore.py --llm3."""
    if not _sem_cache:
        _sem_cache.update(json.load(open(SEMANTIC_CACHE)))
    return _sem_cache[json.dumps([r["gt_text"], r["transcription"] or ""])][f"llm3:{JUDGE}"]


def _num(x, digits=1):
    x = max(x, 0.0)
    return "100" if round(x, digits) == 100 else f"{x:.{digits}f}"


def rate_col(name, success, valid=False, ci=True, higher=True):
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


def asr_columns(full=False):
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


# Bullet: the reference (GATAS on ASR, GATAS_llm on intent) is significantly better than the baseline in this model, task and measure;
# the color gives the size of the paired Cohen's d_z. The LaTeX functions only format the frames.
ALPHA = 0.05
REFERENCE = {"ASR": "GATAS", "Intent": "GATAS_llm"}
OWN = ("GATAS", "GATAS_llm")  # no bullets for variants of the proposed method
BULLET_COLORS = ((1.0, "red!80!black"), (0.5, "orange!90!yellow"), (0.0, "green!50!black"))



def effect_label(d):
    return "large" if d > 1.0 else "medium" if d > 0.5 else "small"


def bullet(d):
    color = next(c for lo, c in BULLET_COLORS if d > lo or lo == 0.0)
    return rf"\,\textcolor{{{color}}}{{$\bullet$}}"


def significance(records, reference, getter, higher, binary):
    """{method: (Holm-adjusted one-sided p, d_z)} of the reference against every other method; d_z > 0: reference better"""
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


def _row(base, recs, getter, scale, test):
    p, d = test if test else (float("nan"), float("nan"))
    better = bool(test and base["method"] not in OWN and p < ALPHA and d > 0)
    return {**base, "n": len(recs), "value": float(np.mean([getter(r) for r in recs])) * scale, "p_holm": p, "d_z": d,
            "ref_better": better, "effect": effect_label(d) if better else ""}


def rq1_measures(task, thr):
    """[(name, LaTeX header, getter, higher is better, binary, digits)] of the RQ1 table for a task; thr = m-valid UTMOS threshold."""
    from results_loader import asr_found, intent_hit
    rate = lambda f: (lambda r: float(f(r)))
    dist = [("WER", r"WER $\downarrow$", lambda r: r["wer"], False, False, 3), ("Semantic", r"Semantic $\downarrow$", lambda r: r["semantic"], False, False, 3),
            ("d_F-Spec", r"$d_F$-Spec $\downarrow$", lambda r: r["spectrogram"], False, False, 3), ("UTMOS", r"UTMOS $\uparrow$", lambda r: r["utmos"], True, False, 2)]
    if task == "ASR":
        return [("Found %", r"Found [\%] $\uparrow$", rate(asr_found), True, True, 1)] + dist + [
            ("M-valid %", r"M-valid [\%] $\uparrow$", rate(lambda r: asr_found(r) and r["utmos"] >= thr), True, True, 1)]
    return [("Target hit %", r"Target hit [\%] $\uparrow$", rate(intent_hit), True, True, 1)] + dist + [
        ("M-valid %", r"M-valid [\%] $\uparrow$", rate(lambda r: intent_hit(r) and r["utmos"] >= thr), True, True, 1)]


def rq1_frame(asr, intent, thr, models=MODELS):
    """RQ1 table values in long format with the Holm-adjusted one-sided p and paired d_z of the reference against each method."""
    rows = []
    for task, data, methods in (("ASR", asr, ASR_ORDER), ("Intent", intent, ("GATAS", "GATAS_llm") + INTENT_ORDER[1:-1])):
        for mo in models:
            for name, _, getter, higher, binary, _ in rq1_measures(task, thr[task]):
                sig = significance(data.get(mo, {}), REFERENCE[task], getter, higher, binary)
                for m in methods:
                    recs = data.get(mo, {}).get(m)
                    if recs:
                        rows.append(_row({"task": task, "model": mo, "method": m, "measure": name}, recs, getter, 100 if binary else 1, sig.get(m)))
    return pd.DataFrame(rows)


def _panel(frame, task, thr, methods, models, names):
    measures = rq1_measures(task, thr)
    f = frame[frame.task == task].set_index(["model", "method", "measure"])
    k = len(measures)
    best = {}
    for mo in models:
        for name, _, _, higher, _, _ in measures:
            vals = [f.loc[(mo, m, name), "value"] for m in methods if (mo, m, name) in f.index]
            if len(set(vals)) > 1:
                best[mo, name] = max(vals) if higher else min(vals)
    rows = []
    for m in methods:
        if not any((mo, m, measures[0][0]) in f.index for mo in models):
            continue
        out = []
        for mo in models:
            for name, _, _, _, binary, digits in measures:
                if (mo, m, name) not in f.index:
                    out.append("--"); continue
                c = f.loc[(mo, m, name)]
                text = "100" if binary and round(c.value, digits) == 100 else f"{c.value:.{digits}f}"
                if (mo, name) in best and abs(c.value - best[mo, name]) < 1e-9:
                    text = rf"\textbf{{{text}}}"
                out.append(text + (bullet(c.d_z) if c.ref_better else ""))
        rows.append(f"{names[m]} & " + " & ".join(out) + r" \\")
    head1 = " & " + " & ".join(rf"\multicolumn{{{k}}}{{c}}{{{MODEL_LABELS[m]}}}" for m in models) + r" \\"
    rules = "".join(rf"\cmidrule(lr){{{2 + g * k}-{1 + (g + 1) * k}}}" for g in range(len(models)))
    head2 = " & " + " & ".join(c[1] for _ in models for c in measures) + r" \\ \midrule"
    return "\n".join([rf"\begin{{tabular}}{{l{'c' * (k * len(models))}}}\toprule", head1, rules, head2, *rows, r"\bottomrule", r"\end{tabular}"])


def rq1_tex(frame, thr, models=MODELS, asr_names=TEX_NAME, intent_names=None):
    """The RQ1 table: panel (a) ASR testing, panel (b) intent matching, formatted from `rq1_frame`."""
    intent_names = intent_names or {**TEX_NAME, "GATAS_llm": r"\gllm"}
    pa = _panel(frame, "ASR", thr["ASR"], ASR_ORDER, models, asr_names)
    pi = _panel(frame, "Intent", thr["Intent"], ("GATAS", "GATAS_llm") + INTENT_ORDER[1:-1], models, intent_names)
    cap = (r"Effectiveness (RQ\textsubscript{1}) on (a) ASR testing and (b) intent matching. Bold: best value. Bullets: significantly better "
           r"than the method (reference: \tool on ASR, \gllm on intent), colored by effect size (\autoref{sec:metrics}). " + NOTE_PGD)
    return "\n".join([r"\begin{table*}[t]", rf"\caption{{{cap}}}", r"\label{tab:rq1}", r"\centering", r"\footnotesize",
                       r"\textbf{(a) ASR testing}\\[2pt]", r"\adjustbox{max width=\textwidth}{%", pa + "}", r"\vspace{6pt}",
                       r"\textbf{(b) Intent matching}\\[2pt]", r"\adjustbox{max width=\textwidth}{%", pi + "}", r"\end{table*}"])


def efficiency_frame(tasks, thr, models=MODELS):
    """Efficiency table values per task, model and method. tasks = {"ASR"/"Intent": {model: {method: records}}} with `runtime` set.
    Queries per m-valid success = mean queries / m-valid success rate; as budgets are nearly equal, it is tested via the m-valid success
    rate (exact McNemar). Runtime: one-sided Wilcoxon."""
    from results_loader import asr_found, intent_hit
    success = {"ASR": asr_found, "Intent": intent_hit}
    rows = []
    for task, data in tasks.items():
        valid = lambda r, t=task: float(success[t](r) and r["utmos"] >= thr[t])
        for mo in models:
            recs_by = data.get(mo, {})
            t_sig = significance(recs_by, REFERENCE[task], lambda r: r["runtime"], False, False)
            q_sig = significance(recs_by, REFERENCE[task], valid, True, True)
            ref_recs = recs_by.get(REFERENCE[task], [])
            ref_rate = float(np.mean([valid(r) for r in ref_recs])) if ref_recs else 0.0
            ref_rows = {"queries_per_m_valid": float(np.mean([r["queries"] for r in ref_recs])) / ref_rate if ref_rate else float("inf")} if ref_recs else {}
            for m, recs in recs_by.items():
                rate = float(np.mean([valid(r) for r in recs]))
                queries = float(np.mean([r["queries"] for r in recs]))
                pt_, dt = t_sig.get(m, (float("nan"), float("nan")))
                pq, dq = q_sig.get(m, (float("nan"), float("nan")))
                own = m in OWN
                ref_q = ref_rows.get("queries_per_m_valid")
                finite = bool(np.isfinite(queries / rate) if rate else False) and ref_q is not None and np.isfinite(ref_q)
                rows.append({"task": task, "model": mo, "method": m, "n": len(recs), "time_s": float(np.mean([r["runtime"] for r in recs])),
                             "queries": queries, "m_valid_rate_%": 100 * rate, "queries_per_m_valid": queries / rate if rate else float("inf"),
                             "p_time": pt_, "d_time": dt, "time_ref_faster": bool(not own and pt_ < ALPHA and dt > 0),
                             "p_m_valid": pq, "d_m_valid": dq, "m_valid_ref_better": bool(not own and finite and pq < ALPHA and dq > 0)})
    return pd.DataFrame(rows)


def efficiency_tex(frame, models=MODELS, names=TEX_NAME):
    """Efficiency table formatted from `efficiency_frame`."""
    tasks = list(dict.fromkeys(frame.task))
    f = frame.set_index(["task", "model", "method"])
    best = {(t, mo): (f.loc[(t, mo)].time_s.min(), f.loc[(t, mo)].queries_per_m_valid.min()) for t in tasks for mo in models if (t, mo) in f.index}

    def fmt(task, model, m):
        if (task, model, m) not in f.index:
            return ["--", "--"]
        c = f.loc[(task, model, m)]
        t = f"{c.time_s:.1f}"
        q = r"$\infty$" if np.isinf(c.queries_per_m_valid) else f"{round(c.queries_per_m_valid):,}".replace(",", r"\,")
        if abs(c.time_s - best[task, model][0]) < 1e-9:
            t = rf"\textbf{{{t}}}"
        if c.time_ref_faster:
            t += bullet(c.d_time)
        if np.isfinite(c.queries_per_m_valid) and abs(c.queries_per_m_valid - best[task, model][1]) < 1e-9:
            q = rf"\textbf{{{q}}}"
        if c.m_valid_ref_better:
            q += bullet(c.d_m_valid)
        return [t, q]

    task_labels = {"ASR": "ASR", "Intent": "Intent (target hit)"}
    g = len(models)
    head1 = " & " + " & ".join(rf"\multicolumn{{{2 * g}}}{{c}}{{{task_labels[t]}}}" for t in tasks) + r" \\"
    rules1 = "".join(rf"\cmidrule(lr){{{2 + i * 2 * g}-{1 + (i + 1) * 2 * g}}}" for i in range(len(tasks)))
    head2 = " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{MODEL_LABELS[mo]}}}" for _ in tasks for mo in models) + r" \\"
    rules2 = "".join(rf"\cmidrule(lr){{{2 + i * 2}-{3 + i * 2}}}" for i in range(len(tasks) * g))
    head3 = " & " + " & ".join(r"Time [s] $\downarrow$ & Q/m-valid $\downarrow$" for _ in range(len(tasks) * g)) + r" \\ \midrule"
    order = [m for m in dict.fromkeys(ASR_ORDER[:1] + INTENT_ORDER[-1:] + ASR_ORDER[1:])]
    rows = []
    for m in order:
        if not any((t, mo, m) in f.index for t in tasks for mo in models):
            continue
        out = [x for t in tasks for mo in models for x in fmt(t, mo, m)]
        rows.append(f"{names[m]} & " + " & ".join(out) + r" \\")
    cols = "l" + "cc" * (len(tasks) * g)
    lines = [r"\begin{table*}[t]", r"\caption{Efficiency (RQ\textsubscript{3}). $\infty$: no m-valid success. Bullets: reference significantly better "
             r"(\autoref{sec:metrics}). " + NOTE_PGD + "}",
             r"\label{tab:efficiency}", r"\centering", r"\footnotesize", r"\adjustbox{max width=\textwidth}{%", rf"\begin{{tabular}}{{{cols}}}\toprule",
             head1, rules1, head2, rules2, head3, *rows, r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    return "\n".join(lines)

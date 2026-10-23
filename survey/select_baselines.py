"""Two best baselines by reciprocal rank fusion (k=60) over the cells (SUT x task) and two measures of the RQ1 table: effectiveness (ASR:
Changed, intent: Target hit) and UTMOS. Ties share the mean rank; a method with several configurations (SMACK) scores with the better one.
Run from the project root: python survey/select_baselines.py
"""
import sys
from collections import defaultdict

sys.path.insert(0, "analysis")
from results_loader import load_results, MODELS
import paper_tables
from paper_tables import asr_columns, intent_columns

paper_tables.SEMANTIC_CACHE = "outputs/semantic_scores.json"

K = 60
ASR = {"Waveform": "Waveform", "SMACK": "SMACK", "SMACK (untargeted)": "SMACK_untargeted", "ALIF": "ALIF", "PGD": "PGD"}
INTENT = {"Waveform": "Waveform_intent", "SMACK": "SMACK", "ALIF": "ALIF_intent", "PGD": "PGD"}
METHOD_OF = {"SMACK (untargeted)": "SMACK"}


def rank(values):
    """Rank 1 = highest value, ties share the mean rank."""
    order = sorted(values, key=values.get, reverse=True)
    ranks = {}
    for m in order:
        tied = [o for o in order if values[o] == values[m]]
        ranks[m] = sum(order.index(o) + 1 for o in tied) / len(tied)
    return ranks


def rankings():
    out = {}
    for sut in MODELS:
        a = load_results("outputs/results", dataset="harvard", asr_model=sut, methods=tuple(ASR.values()), exclude_original_fails=True)
        i = load_results("outputs/results", dataset="commands", asr_model=sut, methods=tuple(INTENT.values()))
        i = {f: [r for r in rs if r["intent"] and r["intent"]["original_correct"]] for f, rs in i.items()}
        tasks = {"ASR": (a, ASR, asr_columns(), ("Changed", "UTMOS")), "Intent": (i, INTENT, intent_columns(), ("Target hit", "UTMOS"))}
        for task, (data, folders, columns, wanted) in tasks.items():
            for name in wanted:  # the column definitions of the RQ1 effectiveness table (paper_tables.py)
                fn = next(c[1] for c in columns if c[0].startswith(name))
                values = {m: fn(data[f], None)[0] for m, f in folders.items() if data[f]}
                out[(task, sut, name)] = (values, rank(values))
    return out


def select(n=2, verbose=False):
    score = defaultdict(float)
    for key, (values, ranks) in rankings().items():
        per_method = defaultdict(float)
        for m, r in ranks.items():
            per_method[METHOD_OF.get(m, m)] = max(per_method[METHOD_OF.get(m, m)], 1 / (K + r))
        for m, v in per_method.items():
            score[m] += v
        if verbose:
            print(key, {m: f"{values[m]:.2f} (#{ranks[m]:g})" for m in values})
    best = sorted(score, key=score.get, reverse=True)
    if verbose:
        print("RRF:", {m: round(score[m], 4) for m in best})
    return best[:n]


if __name__ == "__main__":
    print("best baselines:", select(verbose=True))

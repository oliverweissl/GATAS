"""
Evaluate saved attack results of all methods with the same metrics (src/metrics.py, src/evaluation.py).
Writes evaluation.json next to every summary; the analysis notebooks read only these files.

Run from project root (main env), after the attacks:
    python scripts/evaluate_results.py                          # everything below outputs/results
    python scripts/evaluate_results.py --method GATAS --dataset harvard
    python scripts/evaluate_results.py --overwrite              # recompute existing evaluations
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from glob import glob

import torch
from tqdm import tqdm

from src.evaluation import evaluate, find_summaries
from src.helper import METHODS, RESULTS_DIR
from src.data.datasets import DATASETS
from src.metrics import VALIDITY_REFERENCE_ASR
from src.models import load_asr_model


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved attack results")
    parser.add_argument("--method", type=str, default=None, choices=METHODS, help="Only this method (default: all)")
    parser.add_argument("--dataset", type=str, default=None, choices=DATASETS, help="Only this dataset (default: all)")
    parser.add_argument("--experiment", type=str, default=None, help="Only this experiment folder, e.g. whisper-tiny_20260101_1200")
    parser.add_argument("--overwrite", action="store_true", help="Recompute existing evaluation.json files")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    experiments = glob(os.path.join(RESULTS_DIR, args.method or "*", args.dataset or "*", args.experiment or "*"))
    summaries = [s for e in sorted(experiments) for s in find_summaries(e)]
    if not args.overwrite:
        summaries = [s for s in summaries if not os.path.exists(os.path.join(os.path.dirname(s), "evaluation.json"))]
    print(f"{len(summaries)} results to evaluate in {len(experiments)} experiment folders")
    if not summaries:
        return

    reference_asr = load_asr_model(VALIDITY_REFERENCE_ASR, device=device)
    failed = 0
    for summary in tqdm(summaries):
        try:
            evaluate(summary, reference_asr, device)
        except Exception as e:
            failed += 1
            print(f"[ERROR] {summary}: {e}")
    print(f"[Done] evaluated {len(summaries) - failed}, failed {failed}")


if __name__ == "__main__":
    main()

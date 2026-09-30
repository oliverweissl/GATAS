"""
Uniform evaluation of saved attack results (all methods), used by scripts/evaluate_results.py.

Reads the two summary formats (RunLogger's run_summary.json for GATAS/Waveform, result_writer's
<method>_summary.json for the baselines), recomputes every metric from the saved audio and the attack-time
transcription with src/metrics.py, and writes evaluation.json next to the summary.
"""
import os
import json
from glob import glob

import numpy as np
import soundfile as sf
import torch

from . import metrics

SAMPLE_RATE = 16_000


def find_summaries(experiment_dir: str) -> list:
    paths = glob(os.path.join(experiment_dir, "**", "*summary.json"), recursive=True)
    return sorted(p for p in paths if os.path.basename(p) != "all_results.json")


def normalize_summary(path: str) -> dict:
    """Common record from either summary format."""
    with open(path) as f:
        d = json.load(f)

    if "efficiency_metrics" in d:  # RunLogger (GATAS / Waveform)
        meta, text, eff, algo = d["metadata"], d["text_data"], d["efficiency_metrics"], d["algorithm_parameters"]
        method = meta.get("attack_method")
        return {
            "attack_method": method,
            "dataset": meta.get("dataset"),
            "sentence_id": meta["sentence_id"],
            "run_id": meta.get("run_id") or 0,
            "asr_model": algo.get("asr_model"),
            "gt_text": text["ground_truth_text"],
            # For untargeted modes RunLogger stores the transcription of the noise target here, not a target text
            "target_text": text.get("target_text") if (method or "").endswith("_targeted") else None,
            "transcription": text.get("asr_transcription"),
            "gt_transcription": text.get("gt_transcription"),
            "elapsed_seconds": eff["elapsed_time_seconds"],
            "queries": eff.get("queries", eff["generation_count"] * algo["pop_size"]),
            "generations": eff["generation_count"],
            "pop_size": algo["pop_size"],
            "seed": algo.get("seed"),
            # First generation in which an individual met the objective thresholds (None: never)
            "generation_found": d.get("final_solution", {}).get("generation_found"),
            "pareto_front": d.get("pareto_front", []),
        }

    return {  # result_writer (SMACK / ALIF / PGD)
        "attack_method": d["attack_method"],
        "dataset": d.get("dataset"),
        "sentence_id": d["sentence_id"],
        "run_id": 0,
        "asr_model": d.get("asr_model"),
        "gt_text": d["gt_text"],
        "target_text": d.get("target_text"),
        "transcription": d.get("transcription"),
        "gt_transcription": d.get("gt_transcription"),
        "elapsed_seconds": d["elapsed_seconds"],
        "queries": d["queries"],
        "generations": d["generations"],
        "pop_size": d["pop_size"],
        "seed": d.get("seed"),
        "generation_found": None,
        "pareto_front": [],
    }


def _read(path: str) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        import librosa
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    return audio


def audio_paths(folder: str) -> dict:
    adversarial = sorted(glob(os.path.join(folder, "best_*.wav")))
    clean_reference = os.path.join(folder, "clean_reference.wav")
    ground_truth = os.path.join(folder, "ground_truth.wav")
    return {
        "adversarial": adversarial[0] if adversarial else None,
        "ground_truth": ground_truth,
        # PESQ needs a time-aligned reference of the same voice: methods with their own TTS save one
        "pesq_reference": clean_reference if os.path.exists(clean_reference) else ground_truth,
    }


def text_metrics(gt_text: str, transcription: str, target_text: str = None, device: str = "cpu") -> dict:
    scores = {
        "SET_OVERLAP": metrics.set_overlap(gt_text, transcription),
        "WER": metrics.wer(gt_text, transcription),
        "SBERT_SIMILARITY": metrics.sbert_similarity(gt_text, transcription, device),
    }
    if target_text:
        scores["TARGET_WER"] = metrics.wer(target_text, transcription)
    return scores


def evaluate(summary_path: str, reference_asr, device: str = "cpu") -> dict:
    record = normalize_summary(summary_path)
    folder = os.path.dirname(summary_path)
    paths = audio_paths(folder)
    adversarial = _read(paths["adversarial"])
    ground_truth = _read(paths["ground_truth"])
    pesq_reference = _read(paths["pesq_reference"])

    scores = text_metrics(record["gt_text"], record["transcription"], record["target_text"], device)
    scores["PESQ"] = metrics.pesq_fitness(pesq_reference, adversarial)
    scores["UTMOS"] = metrics.utmos_score(adversarial, device)
    scores["UTMOS_GT"] = metrics.utmos_score(ground_truth, device)

    targeted = record["target_text"] is not None
    thresholds = metrics.TARGETED_SUCCESS_THRESHOLDS if targeted else metrics.SUCCESS_THRESHOLDS
    success = metrics.is_success(scores, thresholds)

    reference_transcription = reference_asr.inference(torch.from_numpy(adversarial)[None, :])[0][0]
    reference_gt_transcription = reference_asr.inference(torch.from_numpy(ground_truth)[None, :])[0][0]
    reference_wer = metrics.wer(record["gt_text"], reference_transcription)
    valid = reference_wer <= metrics.VALIDITY_MAX_WER

    # Pareto front members (GATAS / Waveform): text metrics for post-hoc threshold analysis
    front = []
    for member in record.pop("pareto_front"):
        transcription = member.get("transcription")
        entry = dict(member.get("fitness", {}))
        if transcription is not None:
            entry["WER"] = metrics.wer(record["gt_text"], transcription)
            if targeted:
                entry["TARGET_WER"] = metrics.wer(record["target_text"], transcription)
        entry["transcription"] = transcription
        front.append(entry)

    evaluation = {
        **record,
        "targeted": targeted,
        "summary_path": os.path.abspath(summary_path),
        "audio": {k: os.path.abspath(v) if v else None for k, v in paths.items()},
        "scores": {k: round(float(v), 6) for k, v in scores.items()},
        "success": bool(success),
        "success_thresholds": dict(thresholds),
        "validity": {
            "reference_asr": metrics.VALIDITY_REFERENCE_ASR,
            "reference_transcription": reference_transcription,
            "reference_gt_transcription": reference_gt_transcription,
            "reference_wer": round(reference_wer, 6),
            "reference_gt_wer": round(metrics.wer(record["gt_text"], reference_gt_transcription), 6),
            "max_wer": metrics.VALIDITY_MAX_WER,
            "valid": bool(valid),
        },
        "valid_success": bool(success and valid),
        "pareto_front": front,
    }
    archive_history = os.path.join(folder, "archive_history.json")
    evaluation["archive_history"] = os.path.abspath(archive_history) if os.path.exists(archive_history) else None

    with open(os.path.join(folder, "evaluation.json"), "w") as f:
        json.dump(evaluation, f, indent=2)
    return evaluation

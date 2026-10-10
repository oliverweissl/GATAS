"""
Uniform evaluation of saved attack results (all methods), used by scripts/evaluate_results.py.

Reads the two summary formats (RunLogger's run_summary.json for GATAS/Waveform, result_writer's
<method>_summary.json for the baselines), recomputes the metrics from the saved audio and the attack-time
transcription with src/metrics.py (success, lexical / semantic / spectrogram distance), and writes evaluation.json
next to the summary.
"""
import os
import json
from glob import glob

import numpy as np
import soundfile as sf
import torch

from . import metrics
from .data.datasets import load_intents, load_target_intents
from .intent import classify_intents
from .models import load_asr_model

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
            "target_text": text.get("target_text") if algo.get("attack_mode") == "TARGETED" else None,
            "target_intent": text.get("target_intent"),
            "transcription": text.get("asr_transcription"),
            "elapsed_seconds": eff["elapsed_time_seconds"],
            "queries": eff.get("queries", eff["generation_count"] * algo["pop_size"]),
            "generations": eff["generation_count"],
            "pop_size": algo["pop_size"],
            "seed": algo.get("seed"),
            # First generation in which an individual met the objective thresholds (None: never)
            "generation_found": d.get("final_solution", {}).get("generation_found"),
            "scale": d.get("final_solution", {}).get("scale"),
        }

    return {  # result_writer (SMACK / ALIF / PGD)
        "attack_method": d["attack_method"],
        "dataset": d.get("dataset"),
        "sentence_id": d["sentence_id"],
        "run_id": 0,
        "asr_model": d.get("asr_model"),
        "gt_text": d["gt_text"],
        "target_text": d.get("target_text"),
        "target_intent": d.get("target_intent"),
        "transcription": d.get("transcription"),
        "elapsed_seconds": d["elapsed_seconds"],
        "queries": d["queries"],
        "generations": d["generations"],
        "pop_size": d["pop_size"],
        "seed": d.get("seed"),
        "generation_found": None,
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
    return {
        "adversarial": adversarial[0] if adversarial else None,
        "original": os.path.join(folder, "ground_truth.wav"),
    }


_asr_models = {}


def _transcribe(asr_model_name: str, audio: np.ndarray, device: str) -> str:
    """Transcription of an audio by the attacked ASR model (re-run on the saved audio, not the attack-time text)."""
    if (asr_model_name, device) not in _asr_models:
        _asr_models[(asr_model_name, device)] = load_asr_model(asr_model_name, device=device)
    return _asr_models[(asr_model_name, device)].inference(torch.from_numpy(audio)[None, :])[0][0]


def _intent_evaluation(record: dict, original_transcription: str, device: str):
    """
    Intent matching (datasets with reference intents, e.g. commands): the intent predicted from the transcription.
    The reference intent of a sentence is the classifier's prediction on its clean text (commands.intents.txt), the
    target intent its best other intent of a different scenario (commands.target_intents.txt).
      changed:      the adversarial transcription is classified differently than the original audio's transcription
      target_match: the predicted intent equals the target intent (for every method; `targeted` tells if the method
                    optimized toward it, untargeted methods can hit it by chance)
      original_correct: control, the ASR transcription of the clean audio still maps to the reference intent
    """
    intents = load_intents(record["dataset"]) if record.get("dataset") else None
    if intents is None:
        return None
    reference = intents[record["sentence_id"] - 1]
    target = load_target_intents(record["dataset"])[record["sentence_id"] - 1]
    adversarial_intent, original_intent = classify_intents([record["transcription"], original_transcription], device)
    return {
        "reference": reference,
        "target": target,
        "targeted": record.get("target_intent") is not None,
        "original": original_intent,
        "adversarial": adversarial_intent,
        "original_correct": original_intent == reference,
        "changed": adversarial_intent != original_intent,
        "target_match": adversarial_intent == target,
    }


def evaluate(summary_path: str, device: str = "cpu") -> dict:
    record = normalize_summary(summary_path)
    folder = os.path.dirname(summary_path)
    paths = audio_paths(folder)
    adversarial = _read(paths["adversarial"])
    original = _read(paths["original"])

    # Re-transcribe the saved audio with the attacked ASR model; the attack-time transcription is kept for reference
    attack_transcription = record["transcription"]
    record["transcription"] = _transcribe(record["asr_model"], adversarial, device)
    original_transcription = _transcribe(record["asr_model"], original, device)

    wer = metrics.wer(record["gt_text"], record["transcription"])
    scores = {
        "WER": wer,  # lexical distance
        "SEMANTIC_DISTANCE": metrics.sbert_distance(record["gt_text"], record["transcription"], device),
        "SPECTROGRAM_DISTANCE": metrics.spectrogram_distance(original, adversarial),
        "UTMOS": metrics.utmos_score(adversarial, device),  # naturalness, not optimized by any method
        "UTMOS_ORIGINAL": metrics.utmos_score(original, device),
    }
    evaluation = {
        **record,
        "attack_transcription": attack_transcription,
        "original_transcription": original_transcription,
        # Control: the original audio is transcribed correctly, so a found test case is caused by the attack
        "original_passes": bool(not metrics.is_success(metrics.wer(record["gt_text"], original_transcription))),
        "summary_path": os.path.abspath(summary_path),
        "audio": {k: os.path.abspath(v) if v else None for k, v in paths.items()},
        "scores": {k: round(float(v), 6) for k, v in scores.items()},
        "success": bool(metrics.is_success(wer)),
    }
    evaluation["intent"] = _intent_evaluation(record, original_transcription, device)
    archive_history = os.path.join(folder, "archive_history.json")
    evaluation["archive_history"] = os.path.abspath(archive_history) if os.path.exists(archive_history) else None

    with open(os.path.join(folder, "evaluation.json"), "w") as f:
        json.dump(evaluation, f, indent=2)
    return evaluation

"""
Common result format of the baseline methods (SMACK, ALIF, PGD). GATAS and the waveform baseline write the
richer run_summary.json of RunLogger. Metrics are computed afterwards by scripts/evaluate_results.py, so the
baseline environments only need to save audio and the raw attack facts.

Per sentence (<results>/sentence_XXX/):
  best_<method>.wav      adversarial audio, 16 kHz
  ground_truth.wav       shared reference audio (StyleTTS2), 16 kHz
  clean_reference.wav    optional: benign synthesis of a method with its own TTS (PESQ reference)
  <method>_summary.json  see save_attack_result
"""
import os
import json
import datetime

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16_000


def save_attack_result(
    output_dir: str,
    method: str,
    dataset: str,
    sentence_id: int,
    asr_model: str,
    audio: np.ndarray,
    gt_audio: np.ndarray,
    gt_text: str,
    transcription: str,
    gt_transcription: str,
    elapsed_seconds: float,
    queries: int,
    generations: int,
    pop_size: int,
    seed: int,
    target_text: str = None,
    clean_reference_audio: np.ndarray = None,
    params: dict = None,
) -> dict:
    """
    Args:
        output_dir:        Experiment results directory (src.helper.results_dir).
        method:            Method name (src.helper.METHODS); lowercased in file names.
        asr_model:         Canonical name of the attacked ASR model.
        audio, gt_audio:   Adversarial and ground-truth audio, float in [-1, 1] at 16 kHz.
        transcription:     Attacked-ASR transcription of the adversarial audio.
        gt_transcription:  Attacked-ASR transcription of the ground-truth audio.
        elapsed_seconds:   End-to-end wall clock of the attack.
        queries:           Number of attacked-model queries (PGD: gradient iterations).
        generations, pop_size: Search budget in the method's own terms.
        target_text:       Target transcription for targeted attacks.
        clean_reference_audio: Benign synthesis with the method's own TTS, used as PESQ reference.
        params:            Method-specific hyperparameters.
    """
    folder = os.path.join(output_dir, f"sentence_{sentence_id:03d}")
    os.makedirs(folder, exist_ok=True)
    name = method.lower()

    sf.write(os.path.join(folder, f"best_{name}.wav"), np.asarray(audio, dtype=np.float32).squeeze(), SAMPLE_RATE)
    sf.write(os.path.join(folder, "ground_truth.wav"), np.asarray(gt_audio, dtype=np.float32).squeeze(), SAMPLE_RATE)
    if clean_reference_audio is not None:
        sf.write(os.path.join(folder, "clean_reference.wav"), np.asarray(clean_reference_audio, dtype=np.float32).squeeze(), SAMPLE_RATE)

    summary = {
        "attack_method": method,
        "dataset": dataset,
        "sentence_id": sentence_id,
        "asr_model": asr_model,
        "timestamp": datetime.datetime.now().isoformat(),
        "gt_text": gt_text,
        "target_text": target_text,
        "transcription": transcription,
        "gt_transcription": gt_transcription,
        "elapsed_seconds": round(float(elapsed_seconds), 2),
        "queries": int(queries),
        "generations": int(generations),
        "pop_size": int(pop_size),
        "seed": seed,
        "params": params or {},
    }
    with open(os.path.join(folder, f"{name}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[{sentence_id:3d}] Saved best_{name}.wav | transcription: {transcription!r}")
    return summary

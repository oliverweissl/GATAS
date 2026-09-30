import os
import json

import torch
import soundfile as sf


def save_attack_result(
    output_dir: str,
    sentence_id: int,
    method: str,
    audio,
    transcription: str,
    gt_text: str,
    elapsed: float,
    params: dict,
    evaluation: dict = None,
    gt_audio=None,
    naturalness: dict = None,
    gt_transcription: str = None,
) -> None:
    """
    Save adversarial audio, the ground truth and a standardised JSON for one attack result
    into <output_dir>/sentence_XXX/ as best_<method>.wav, ground_truth.wav and <method>_summary.json (method lowercased).

    Args:
        output_dir:    Experiment results directory (src.helper.results_dir).
        sentence_id:   1-based Harvard sentence index.
        method:        Method name (src.helper.METHODS), lowercased in the file names.
        audio:         Audio as a numpy array or torch.Tensor.
        transcription: ASR transcription of the adversarial audio.
        gt_text:       Original Harvard sentence text.
        elapsed:       End-to-end wall-clock seconds the attack took.
        params:        Method-specific hyperparameters dict.
        evaluation:    Output of src.metrics.evaluate_attack (shared metrics + success).
        gt_audio:      Ground-truth audio (16 kHz), saved as ground_truth.wav.
        gt_transcription: ASR transcription of the ground-truth audio.
        naturalness:   {'utmos_best': ..., 'utmos_gt': ...}.
    """
    sentence_dir = os.path.join(output_dir, f'sentence_{sentence_id:03d}')
    os.makedirs(sentence_dir, exist_ok=True)

    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy().squeeze()

    sf.write(os.path.join(sentence_dir, f'best_{method.lower()}.wav'), audio, samplerate=16000)
    if gt_audio is not None:
        sf.write(os.path.join(sentence_dir, 'ground_truth.wav'), gt_audio, samplerate=16000)

    with open(os.path.join(sentence_dir, f'{method.lower()}_summary.json'), 'w') as f:
        json.dump({
            'attack_method': method,
            'sentence_id': sentence_id,
            'gt_text': gt_text,
            'gt_transcription': gt_transcription,
            'transcription': transcription,
            'elapsed_seconds': round(elapsed, 2),
            'params': params,
            'success_metrics': {
                'success': evaluation['success'],
                'metric_scores': evaluation['scores'],
                'thresholds': evaluation['thresholds'],
            } if evaluation else None,
            'naturalness_scores': naturalness,
        }, f, indent=2)

    print(f"[{sentence_id:3d}] Saved best_{method.lower()}.wav | transcription: {transcription!r}")
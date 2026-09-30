"""
Shared evaluation metrics and success criterion for all attack methods (GATAS, Waveform, SMACK, PGD).

Change SUCCESS_THRESHOLDS to change what counts as a successful attack everywhere.
Both metrics are fitness values where lower is better for the attacker.
"""
import re

import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

SAMPLE_RATE = 16_000

# An attack succeeds if every metric is <= its threshold.
SUCCESS_THRESHOLDS = {
    "PESQ": 0.2,         # PESQ fitness: 0.0 = perfect quality, 1.0 = worst
    "SET_OVERLAP": 0.5,  # fraction of GT content words surviving in the transcription
}

_STOPWORDS = set(stopwords.words('english'))
_LEMMATIZER = WordNetLemmatizer()


def _lemmatize_word(word: str) -> str:
    # Try POS tags in order adjective, verb, noun, adverb and return the first lemma that
    # changes the word ("smoothest" -> "smooth", "slid" -> "slide", "bananas" -> "banana").
    for pos in ('a', 'v', 'n', 'r'):
        lemma = _LEMMATIZER.lemmatize(word, pos=pos)
        if lemma != word:
            return lemma
    return word


def content_lemmas(text: str) -> set[str]:
    """Lowercase, strip punctuation, drop stopwords and lemmatize."""
    clean = re.sub(r'[^\w\s]', '', (text or '').lower())
    return {_lemmatize_word(w) for w in set(clean.split()) - _STOPWORDS}


def set_overlap(gt_text: str, asr_text: str, gt_lemmas: set[str] = None) -> float:
    """
    Fraction of GT content-word lemmas that survive in the ASR transcription.
    0.0 = no GT content word left (attack success), 1.0 = all GT content words present.
    """
    if gt_lemmas is None:
        gt_lemmas = content_lemmas(gt_text)
    if not gt_lemmas:
        return 0.0
    if not asr_text:
        return 0.0
    return min(len(gt_lemmas & content_lemmas(asr_text)) / len(gt_lemmas), 1.0)


def pesq_fitness(reference_audio: np.ndarray, degraded_audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    """
    Wideband PESQ mapped to a fitness in [0, 1]: 0.0 = perfect (PESQ 4.5), 1.0 = worst (PESQ -0.5).
    PESQ failures (e.g. silent audio) count as worst quality.
    """
    from pesq import pesq

    try:
        score = pesq(sample_rate, np.asarray(reference_audio).squeeze(), np.asarray(degraded_audio).squeeze(), 'wb')
    except Exception:
        score = -0.5
    return float(max(0.0, min(1.0, 1.0 - (score + 0.5) / 5.0)))


def is_success(scores: dict) -> bool:
    """scores: {metric name: fitness value} containing every key in SUCCESS_THRESHOLDS."""
    return all(scores[name] <= threshold for name, threshold in SUCCESS_THRESHOLDS.items())


def evaluate_attack(reference_audio: np.ndarray, adversarial_audio: np.ndarray, gt_text: str, asr_text: str) -> dict:
    """Compute the shared metrics and success flag for one adversarial example (16 kHz audio)."""
    scores = {
        "PESQ": round(pesq_fitness(reference_audio, adversarial_audio), 6),
        "SET_OVERLAP": round(set_overlap(gt_text, asr_text), 6),
    }
    return {
        "success": is_success(scores),
        "scores": scores,
        "thresholds": dict(SUCCESS_THRESHOLDS),
    }


# ---------------------------------------------------------------------------
# Naturalness: UTMOS (reported for all methods, not part of the success criterion)
# ---------------------------------------------------------------------------

_utmos_models = {}


def load_utmos(device: str = "cpu"):
    """UTMOS (sarulab UTMOS22 strong learner) traced to TorchScript, from huggingface.co/balacoon/utmos."""
    if device not in _utmos_models:
        import torch
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download(repo_id="balacoon/utmos", filename="utmos.jit", repo_type="model")
        _utmos_models[device] = torch.jit.load(model_path, map_location=device).eval()
    return _utmos_models[device]


def utmos_scores(audio_batch, device: str = "cpu") -> list[float]:
    """
    Predicted MOS in [1, 5] for 16 kHz audio (float in [-1, 1]); audio_batch: [Batch, Time] or [Time].
    The traced model expects int16 samples.
    """
    import torch

    audio = torch.as_tensor(np.asarray(audio_batch) if not isinstance(audio_batch, torch.Tensor) else audio_batch)
    audio = audio.detach().float()
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    audio_int16 = (audio.clamp(-1.0, 1.0) * 32767.0).round().to(torch.int16).to(device)
    with torch.no_grad():
        mos = load_utmos(device)(audio_int16)
    return [float(v) for v in mos.flatten().cpu()]


def utmos_score(audio, device: str = "cpu") -> float:
    """Predicted MOS in [1, 5] for one 16 kHz waveform."""
    return round(utmos_scores(audio, device)[0], 4)

"""
Shared metrics for all attack methods (GATAS, Waveform, SMACK, ALIF, PGD); scripts/evaluate_results.py applies the
evaluation ones to every saved result.

Evaluation (ground truth text / audio vs. attacked transcription / audio):
  success               transcription differs from the ground truth text: WER > 0
  lexical distance      WER (word level edit distance)
  semantic distance     1 - cosine similarity of SBERT embeddings
  spectrogram distance  normalized Frobenius distance of the mel spectrograms (dB)
  naturalness           UTMOS of the adversarial audio (and of the ground truth as reference)
Metrics that are optimization objectives (PESQ, SET_OVERLAP) are deliberately not used for evaluation.
"""
import re

import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

SAMPLE_RATE = 16_000

SBERT_MODEL = "all-MiniLM-L6-v2"

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


def is_success(wer_score: float) -> bool:
    """Test case found: the transcription differs from the ground truth text."""
    return wer_score > 0


_WER_TRANSFORM = None


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate after lowercasing, contraction expansion and punctuation removal (can exceed 1)."""
    import jiwer

    global _WER_TRANSFORM
    if _WER_TRANSFORM is None:
        _WER_TRANSFORM = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.ExpandCommonEnglishContractions(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.ReduceToListOfListOfWords(),
        ])
    if not (hypothesis or "").strip():
        return 1.0
    return float(jiwer.wer(reference, hypothesis, reference_transform=_WER_TRANSFORM, hypothesis_transform=_WER_TRANSFORM))


_sbert_models = {}


def sbert_distance(text_a: str, text_b: str, device: str = "cpu") -> float:
    """Semantic distance: 1 - cosine similarity of sentence embeddings (0 = same meaning)."""
    from sentence_transformers import SentenceTransformer, util

    if device not in _sbert_models:
        _sbert_models[device] = SentenceTransformer(SBERT_MODEL, device=device)
    embeddings = _sbert_models[device].encode([text_a or "", text_b or ""], convert_to_tensor=True)
    return 1.0 - float(util.cos_sim(embeddings[0], embeddings[1]).item())


def spectrogram_distance(reference_audio: np.ndarray, degraded_audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    """Frobenius norm of the mel spectrogram (dB) difference, zero-padded to equal length, divided by the spectrogram size."""
    import librosa

    def mel_db(audio):
        mel = librosa.feature.melspectrogram(y=np.asarray(audio).squeeze(), sr=sample_rate, n_fft=1024, hop_length=256, n_mels=128)
        return librosa.power_to_db(np.abs(mel))

    a, b = mel_db(reference_audio), mel_db(degraded_audio)
    frames = max(a.shape[1], b.shape[1])
    a = np.pad(a, ((0, 0), (0, frames - a.shape[1])))
    b = np.pad(b, ((0, 0), (0, frames - b.shape[1])))
    return float(np.linalg.norm(a - b, ord="fro") / a.size)


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

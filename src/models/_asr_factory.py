from __future__ import annotations


# Canonical ASR model name -> (wrapper, checkpoint). The canonical name is used in result folder names.
ASR_MODELS = {
    "whisper-tiny": ("whisper", "tiny"),
    "whisper-large-v3-turbo": ("whisper", "large-v3-turbo"),
    "whisper-large-v3": ("whisper", "large-v3"),  # reference ASR for the validity check (src/metrics.py)
    "wav2vec2-large": ("wav2vec2", "facebook/wav2vec2-large-960h-lv60-self"),
    "wav2vec2-base": ("wav2vec2", "facebook/wav2vec2-base-960h"),
    "speechbrain": ("speechbrain", "speechbrain/asr-transformer-transformerlm-librispeech"),
}

ASR_MODEL_ALIASES = {
    "whisper": "whisper-tiny",
    "wav2vec2": "wav2vec2-large",
}

ASR_MODEL_CHOICES = tuple(ASR_MODELS) + tuple(ASR_MODEL_ALIASES)


def canonical_asr_model_name(name: str | None) -> str:
    key = (name or "whisper-tiny").strip().lower()
    key = ASR_MODEL_ALIASES.get(key, key)
    if key not in ASR_MODELS:
        raise ValueError(f"Unsupported ASR model '{name}'. Choose one of: {', '.join(ASR_MODEL_CHOICES)}")
    return key


def load_asr_model(name: str | None = "whisper-tiny", device: str | None = None):
    wrapper, checkpoint = ASR_MODELS[canonical_asr_model_name(name)]
    if wrapper == "whisper":
        from ._whisper import Whisper

        return Whisper(model_name=checkpoint, device=device)
    if wrapper == "wav2vec2":
        from ._wav2vec2 import Wav2Vec2ASR

        return Wav2Vec2ASR(model_id=checkpoint, device=device)
    if wrapper == "speechbrain":
        from ._speechbrain_asr import SpeechBrainASR

        return SpeechBrainASR(model_id=checkpoint, device=device)

    raise AssertionError(f"Unhandled ASR wrapper: {wrapper}")

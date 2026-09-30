from __future__ import annotations


ASR_MODEL_ALIASES = {
    "whisper": "whisper",
    "whisperasr": "whisper",
    "wav2vec2": "wav2vec2",
    "wav2vec2asr": "wav2vec2",
    "w2v2": "wav2vec2",
    "speechbrain": "speechbrain",
    "speechbrainasr": "speechbrain",
}

ASR_MODEL_CHOICES = ("whisper", "wav2vec2", "speechbrain")


def canonical_asr_model_name(name: str | None) -> str:
    key = (name or "whisper").replace("_", "").replace("-", "").lower()
    try:
        return ASR_MODEL_ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported ASR model '{name}'. Choose one of: {', '.join(ASR_MODEL_CHOICES)}"
        ) from exc


def load_asr_model(name: str | None = "whisper", device: str | None = None):
    canonical = canonical_asr_model_name(name)
    if canonical == "whisper":
        from ._whisper import Whisper

        return Whisper(device=device)
    if canonical == "wav2vec2":
        from ._wav2vec2 import Wav2Vec2ASR

        return Wav2Vec2ASR(device=device)
    if canonical == "speechbrain":
        from ._speechbrain_asr import SpeechBrainASR

        return SpeechBrainASR(device=device)

    raise AssertionError(f"Unhandled ASR model: {canonical}")

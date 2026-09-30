from __future__ import annotations

import torch
from transformers import AutoModelForCTC, AutoProcessor

from ._asr_audio import SAMPLE_RATE, audio_to_numpy_list, clean_transcripts


class Wav2Vec2ASR:
    """Hugging Face CTC ASR wrapper for wav2vec 2.0 checkpoints."""

    def __init__(
        self,
        model_id: str = "facebook/wav2vec2-base-960h",
        device: str | None = None,
    ):
        self.model_id = model_id
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForCTC.from_pretrained(model_id).to(self.device).eval()

    @torch.no_grad()
    def inference(self, audio_batch, sample_rate: int = SAMPLE_RATE):
        audio_arrays = audio_to_numpy_list(
            audio_batch,
            sample_rate=sample_rate,
            target_sample_rate=SAMPLE_RATE,
        )
        if not audio_arrays:
            return [], None

        inputs = self.processor(
            audio_arrays,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        logits = self.model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        texts = self.processor.batch_decode(predicted_ids)

        return clean_transcripts(texts), logits

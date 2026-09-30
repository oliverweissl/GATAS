from __future__ import annotations

import os
import re

import torch
from torch.nn.utils.rnn import pad_sequence

from ._asr_audio import SAMPLE_RATE, audio_to_tensor_list, clean_transcripts


class SpeechBrainASR:
    """SpeechBrain pretrained ASR wrapper."""

    def __init__(
        self,
        model_id: str = "speechbrain/asr-transformer-transformerlm-librispeech",
        device: str | None = None,
        savedir: str | None = None,
    ):
        self.model_id = model_id
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        try:
            from speechbrain.inference.ASR import EncoderDecoderASR
        except ImportError:
            from speechbrain.pretrained import EncoderDecoderASR

        if savedir is None:
            safe_model_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_id)
            savedir = os.path.join("checkpoints", "speechbrain", safe_model_id)

        self.model = EncoderDecoderASR.from_hparams(
            source=model_id,
            savedir=savedir,
            run_opts={"device": self.device},
        )

    @torch.no_grad()
    def inference(self, audio_batch, sample_rate: int = SAMPLE_RATE):
        wavs = audio_to_tensor_list(
            audio_batch,
            sample_rate=sample_rate,
            target_sample_rate=SAMPLE_RATE,
            device=self.device,
        )
        if not wavs:
            return [], None

        lengths = torch.tensor([wav.numel() for wav in wavs], device=self.device, dtype=torch.float32)
        padded = pad_sequence(wavs, batch_first=True)
        wav_lens = lengths / lengths.max()

        result = self.model.transcribe_batch(padded, wav_lens)
        hypotheses = result[0] if isinstance(result, tuple) else result

        texts = []
        for hyp in hypotheses:
            if isinstance(hyp, str):
                texts.append(hyp)
            elif isinstance(hyp, (list, tuple)):
                texts.append(" ".join(str(word) for word in hyp))
            else:
                texts.append(str(hyp))

        return clean_transcripts(texts), None

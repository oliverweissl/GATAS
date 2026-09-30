from __future__ import annotations

from pathlib import Path
import string
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as taf


SAMPLE_RATE = 16_000


def clean_transcripts(texts: list[str]) -> list[str]:
    table = str.maketrans("", "", string.punctuation)
    return [str(text or "").translate(table).strip() for text in texts]


def _numpy_to_tensor(audio: np.ndarray) -> torch.Tensor:
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if np.issubdtype(audio.dtype, np.integer):
        max_value = max(abs(np.iinfo(audio.dtype).min), np.iinfo(audio.dtype).max)
        audio = audio.astype(np.float32) / float(max_value)
    else:
        audio = audio.astype(np.float32)

    return torch.from_numpy(audio)


def _torch_to_batch(audio: torch.Tensor) -> list[torch.Tensor]:
    if audio.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        max_value = max(abs(torch.iinfo(audio.dtype).min), torch.iinfo(audio.dtype).max)
        audio = audio.float() / float(max_value)
    else:
        audio = audio.detach().float()

    audio = audio.cpu()
    while audio.dim() > 2 and 1 in audio.shape:
        audio = audio.squeeze(audio.shape.index(1))

    if audio.dim() == 1:
        return [audio]
    if audio.dim() == 2:
        if audio.shape[1] <= 8 and audio.shape[0] > audio.shape[1]:
            return [audio.mean(dim=1)]
        return [item for item in audio]

    raise ValueError(f"Unsupported audio tensor shape: {tuple(audio.shape)}")


def audio_to_tensor_list(
    audio: Any,
    sample_rate: int = SAMPLE_RATE,
    target_sample_rate: int = SAMPLE_RATE,
    device: str | torch.device | None = None,
) -> list[torch.Tensor]:
    if isinstance(audio, (str, Path)):
        data, source_sample_rate = sf.read(str(audio), dtype="float32", always_2d=False)
        tensors = [_numpy_to_tensor(data)]
    elif isinstance(audio, np.ndarray):
        source_sample_rate = sample_rate
        tensors = [_numpy_to_tensor(audio)]
    elif isinstance(audio, torch.Tensor):
        source_sample_rate = sample_rate
        tensors = _torch_to_batch(audio)
    else:
        raise TypeError(f"Unsupported audio input type: {type(audio)!r}")

    normalized = []
    for wav in tensors:
        if source_sample_rate != target_sample_rate:
            wav = taf.resample(wav.unsqueeze(0), source_sample_rate, target_sample_rate).squeeze(0)
        normalized.append(wav.to(device) if device is not None else wav)

    return normalized


def audio_to_numpy_list(
    audio: Any,
    sample_rate: int = SAMPLE_RATE,
    target_sample_rate: int = SAMPLE_RATE,
) -> list[np.ndarray]:
    tensors = audio_to_tensor_list(
        audio,
        sample_rate=sample_rate,
        target_sample_rate=target_sample_rate,
        device=None,
    )
    return [wav.cpu().numpy() for wav in tensors]

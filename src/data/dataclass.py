from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import torch
from .enum import AttackMode

@dataclass
class ConfigData:
    """
    Central storage for all static run settings.
    Pass this object instead of the 'data' dictionary.
    """
    # --- User Inputs ---
    text_gt: str
    text_target: str

    # --- Runtime Settings ---
    num_generations: int
    pop_size: int
    loop_count: int
    iv_scalar: float
    rank_text: int
    rank_style: int
    batch_size: int
    asr_model_name: str

    # --- Logic Constants ---
    mode: AttackMode
    active_objectives: List["FitnessObjective"]
    thresholds: Dict["FitnessObjective", float]

    # --- Target Selection ---
    num_rms_candidates: int = 20
    adaptive_scale: bool = False
    basis: str = "pca"  # "pca" | "jacobian" (directions ranked by audible effect)
    text_dct: int = 0  # >0: text coefficients are this many DCT components along the phoneme axis
    sens_phonemes: int = 8  # jacobian basis: phonemes probed for the text sensitivity
    target_intent: Optional[str] = None  # intent matching: target intent of the INTENT_* objective
    intent_selection: bool = False  # intent matching: select the final candidate by the intent objective first

@dataclass
class ModelData:
    tts_model: Any  # StyleTTS2
    asr_model: Any  # AutomaticSpeechRecognitionModel

@dataclass
class ObjectiveContext:
    """
    Simplified context for objective evaluation.
    Contains only the data needed by objectives.
    """
    audio_mixed_batch: torch.Tensor  # [Batch, Time]
    asr_texts: list[str]
    interpolation_vectors: torch.Tensor  # [Batch, Dim]
    mel_batch: Optional[torch.Tensor] = None  # [Batch, n_mels, Time]

    def get_item(self, index: int) -> "ObjectiveContext":
        """Extract a single-item context from a batch."""
        return ObjectiveContext(
            audio_mixed_batch=self.audio_mixed_batch[index],
            asr_texts=self.asr_texts[index],
            interpolation_vectors=self.interpolation_vectors[index],
            mel_batch=self.mel_batch[index:index+1] if self.mel_batch is not None else None
        )

    def __len__(self) -> int:
        return len(self.asr_texts)


@dataclass
class AudioEmbeddingData:
    input_length: torch.Tensor
    text_mask: torch.Tensor

    h_bert: torch.Tensor
    h_text: torch.Tensor

    style_vector_acoustic: torch.Tensor
    style_vector_prosodic: torch.Tensor

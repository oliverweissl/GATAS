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
    size_per_phoneme: int
    batch_size: int
    asr_model_name: str

    # --- Logic Constants ---
    mode: AttackMode
    active_objectives: List["FitnessObjective"]
    thresholds: Dict["FitnessObjective", float]

    # --- Subspace Optimization ---
    subspace_optimization: bool
    random_matrix: torch.Tensor

    # --- Target Selection ---
    num_rms_candidates: int = 20

@dataclass
class ModelData:
    # Required Audio
    tts_model: Any  # StyleTTS2
    asr_model: Any  # AutomaticSpeechRecognitionModel

    # Conditional Audio (Default to None)
    embedding_model: Optional[Any] = None  # SentenceTransformer (MPNet)
    sbert_model: Optional[Any] = None  # SentenceTransformer (MiniLM)

    # JIWER Transformation Object
    wer_transformations: Optional[Any] = None

    # GPT-2 / Perplexity
    perplexity_model: Optional[Any] = None  # GPT2LMHeadModel
    perplexity_tokenizer: Optional[Any] = None  # GPT2Tokenizer

    # Wav2Vec2
    wav2vec_model: Optional[Any] = None  # Wav2Vec2Model
    wav2vec_processor: Optional[Any] = None  # Wav2Vec2Processor

@dataclass
class ModelEmbeddingData:
    """
    Dynamic container for conditional reference embeddings.
    Initialize with EmbeddingData() to start empty.
    """
    # Sentence Transformer Embeddings (MPNet)
    text_embedding_gt: Optional[torch.Tensor] = None
    text_embedding_target: Optional[torch.Tensor] = None

    # SBERT Embeddings (MiniLM)
    s_bert_embedding_gt: Optional[torch.Tensor] = None
    s_bert_embedding_target: Optional[torch.Tensor] = None

    # Wav2Vec2 Embeddings
    wav2vec_embedding_gt: Optional[torch.Tensor] = None
    wav2vec_embedding_target: Optional[torch.Tensor] = None

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

from ..base_objective import BaseObjective
from ...data.dataclass import ObjectiveContext
from ...metrics import wer


class WerTargetObjective(BaseObjective):
    """
    Word Error Rate between ASR text and target text.

    WER = (Substitutions + Deletions + Insertions) / Number_of_reference_words
    Values: usually (0, 1), rarely > 1
    0 = perfect match, 1 = 100% of words wrong

    Lower is better (we want ASR output to match target).
    Output is normalized to (0, 1) where:
         0 = 100% similarity / matches target (good for attack)
         1 = 0% similarity / different from target (bad for attack)
    """
    @property
    def supports_batching(self) -> bool:
        return True

    def _calculate_logic(self, context: ObjectiveContext) -> list[float]:
        """Batched WER (src/metrics.py): 0 = matches target (good), up to 2 = different (bad)."""
        return [min(2.0, wer(self.text_target, asr_text)) for asr_text in context.asr_texts]

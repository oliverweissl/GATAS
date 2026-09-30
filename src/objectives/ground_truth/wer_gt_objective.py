from ..base_objective import BaseObjective
from ...data.dataclass import ObjectiveContext
from ...metrics import wer


class WerGtObjective(BaseObjective):
    """
    Word Error Rate between ASR text and ground-truth text (inverted).

    WER = (Substitutions + Deletions + Insertions) / Number_of_reference_words
    Values: usually (0, 1), rarely > 1
    0 = perfect match, 1 = 100% of words wrong

    We INVERT this: higher WER (more different from GT) is better.
    Output is normalized to (0, 1) where:
         0 = 0% similarity / very different from GT (good for attack)
         1 = 100% similarity / same as GT (bad for attack)
    """
    @property
    def supports_batching(self) -> bool:
        return True

    def _calculate_logic(self, context: ObjectiveContext) -> list[float]:
        """Batched WER (src/metrics.py): 0 = different from GT (good), 1 = same as GT (bad)."""
        return [1.0 - min(1.0, wer(self.text_gt, asr_text)) for asr_text in context.asr_texts]

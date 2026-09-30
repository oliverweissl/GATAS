from ..base_objective import BaseObjective
from ...data.dataclass import ObjectiveContext
from ...metrics import content_lemmas, set_overlap


class SetOverlapObjective(BaseObjective):
    """
    Calculates the percentage of Ground Truth content words that 'survived' in the ASR output.

    Pre-processing pipeline (applied to both GT and ASR, see src/metrics.py):
      1. Lowercase + strip punctuation
      2. Remove stopwords (function words that appear regardless of distortion)
      3. WordNet-lemmatize, trying adjective, verb, noun, adverb POS in that order
         ("smoothest" → "smooth", "slid" → "slide", "bananas" → "banana")

    Formula: Intersection(lemma(GT_content), lemma(ASR_content)) / len(lemma(GT_content))

    Range:
        0.0 = SUCCESS (No original content word lemmas found in ASR output)
        1.0 = FAILURE (All original content word lemmas found)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Pre-compute lemmatized Ground Truth content word set
        self.gt_words_set = content_lemmas(self.text_gt)

    @property
    def supports_batching(self) -> bool:
        return True

    def _calculate_logic(self, context: ObjectiveContext) -> list[float]:
        return [set_overlap(self.text_gt, asr_text, gt_lemmas=self.gt_words_set) for asr_text in context.asr_texts]

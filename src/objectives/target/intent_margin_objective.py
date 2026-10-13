import numpy as np

from ..base_objective import BaseObjective
from ...data.dataclass import ObjectiveContext
from ...intent import target_margin


class IntentMarginObjective(BaseObjective):
    """
    Intent matching, margin formulation: sigmoid(-margin / MARGIN_SCALE) with margin = logit(target intent) - largest other
    logit of the intent classifier on the ASR transcription.

    Unlike 1 - P(target) it does not saturate near 1 for transcriptions far from the target, so the search keeps a gradient
    of preference there. 0.5 is exactly the decision boundary: a threshold of 0.5 means that the target intent is the
    predicted intent, which is what the evaluation counts as target_match.
    """

    MARGIN_SCALE = 4.0

    @property
    def supports_batching(self) -> bool:
        return True

    def _calculate_logic(self, context: ObjectiveContext) -> list[float]:
        margins = np.asarray(target_margin(context.asr_texts, self.target_intent, self.device))
        return (1.0 / (1.0 + np.exp(margins / self.MARGIN_SCALE))).tolist()

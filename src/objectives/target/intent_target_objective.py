from ..base_objective import BaseObjective
from ...data.dataclass import ObjectiveContext
from ...intent import target_probability


class IntentTargetObjective(BaseObjective):
    """
    Intent matching: probability that the intent classifier assigns the target intent to the ASR transcription.
    The system under test is ASR + intent classifier; its output is a graded black-box signal.

    Output is 1 - P(target intent): 0 = the SUT outputs the target intent with certainty (good for the attack),
    1 = target intent has probability 0. A threshold of 0.5 means that the target intent is more likely than not.
    """

    @property
    def supports_batching(self) -> bool:
        return True

    def _calculate_logic(self, context: ObjectiveContext) -> list[float]:
        return [1.0 - p for p in target_probability(context.asr_texts, self.target_intent, self.device)]

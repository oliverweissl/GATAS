from enum import Enum

from .naturalness.pesq_objective import PesqObjective
from .target.wer_target_objective import WerTargetObjective
from .target.intent_target_objective import IntentTargetObjective
from .target.intent_margin_objective import IntentMarginObjective
from .ground_truth.set_overlap_objective import SetOverlapObjective


class FitnessObjective(Enum):
    """Objectives of the search (all minimized, order = column order of the scores)."""
    PESQ = PesqObjective  # naturalness: distance to the reference audio
    WER_TARGET = WerTargetObjective  # targeted: transcription vs. target text
    INTENT_TARGET = IntentTargetObjective  # intent matching: 1 - P(target intent | transcription)
    INTENT_MARGIN = IntentMarginObjective  # intent matching: sigmoid(-(target logit - best other logit) / 4), 0.5 = target is predicted
    SET_OVERLAP = SetOverlapObjective  # untargeted: content words of the ground truth that survive

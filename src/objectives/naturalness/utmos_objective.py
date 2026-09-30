from ..base_objective import BaseObjective
from ...data.dataclass import ObjectiveContext
from ...metrics import load_utmos, utmos_score


class UtmosObjective(BaseObjective):
    """UTMOS naturalness (src/metrics.py) as fitness: 0.0 = MOS 5 (best), 1.0 = MOS 1 (worst)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print(f"[INFO] Loading UTMOS Model on {self.device}...")
        load_utmos(self.device)

    @property
    def supports_batching(self):
        # The traced model is fed one waveform at a time; BaseObjective loops over the batch.
        return False

    def _calculate_logic(self, context: ObjectiveContext) -> float:
        mos = utmos_score(context.audio_mixed_batch, self.device)
        return max(0.0, min(1.0, 1.0 - (mos - 1.0) / 4.0))

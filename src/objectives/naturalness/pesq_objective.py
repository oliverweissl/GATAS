from ..base_objective import BaseObjective
from ...data.dataclass import ObjectiveContext
from ...metrics import pesq_fitness


class PesqObjective(BaseObjective):
    """Wideband PESQ against the ground-truth audio, as fitness: 0.0 = perfect, 1.0 = worst."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Cache GT audio as numpy (computed on first use)
        self.cached_gt_16k = None

    @property
    def supports_batching(self):
        """
        PESQ is a CPU library (C++ binding) that accepts single numpy arrays.
        It cannot handle GPU tensors or batches natively.
        We return False so BaseObjective loops for us.
        """
        return False

    def _calculate_logic(self, context: ObjectiveContext) -> float:
        """
        Calculates PESQ for a SINGLE candidate (Not Batched).
        """
        if self.cached_gt_16k is None:
            self.cached_gt_16k = self.audio_gt.squeeze().cpu().numpy()

        return pesq_fitness(self.cached_gt_16k, context.audio_mixed_batch.squeeze().cpu().numpy())

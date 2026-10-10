from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from .base_optimizer import Optimizer
from .optimizer_candidate import OptimizerCandidate


class _CmaBase(Optimizer):
    """Shared state of the CMA-ES optimizers: genome in [bounds], fixed population size, Pareto archive of the base class."""

    def __init__(self, bounds: tuple[int, int], pop_size: int, num_objectives: int, solution_shape: tuple[int, ...]) -> None:
        super().__init__(num_objectives)
        self._bounds = bounds
        self._pop_size = pop_size
        self._shape = solution_shape
        self._n_var = int(np.prod(solution_shape))
        self._best_candidates = [
            OptimizerCandidate(solution=np.random.uniform(*bounds, size=self._n_var), fitness=[np.inf] * num_objectives)
        ]

    def get_x_current(self) -> NDArray:
        return self._x_current.reshape((self._x_current.shape[0], *self._shape))


class CmaesOptimizer(_CmaBase):
    """
    (mu/mu_w, lambda)-CMA-ES (pycma) on the Chebyshev scalarization max_i f_i / threshold_i of the objectives.
    A value <= 1 means that every threshold is met, so the scalar is minimized until the test case is found.
    `weights` are 1 / threshold (1 for objectives without a positive threshold).
    Diagonal covariance by default: the genome has hundreds of genes and the budget is a few hundred evaluations.
    """

    def __init__(self, bounds: tuple[int, int], pop_size: int, num_objectives: int, solution_shape: tuple[int, ...],
                 weights: Sequence[float], sigma0: float = 0.3, diagonal: bool = True) -> None:
        import cma

        super().__init__(bounds, pop_size, num_objectives, solution_shape)
        self._weights = np.asarray(weights, dtype=float)
        center = 0.5 * (bounds[0] + bounds[1])
        self._es = cma.CMAEvolutionStrategy(
            [center] * self._n_var, sigma0 * (bounds[1] - bounds[0]),
            {"bounds": list(bounds), "popsize": pop_size, "CMA_diagonal": diagonal, "verbose": -9,
             "seed": int(np.random.randint(1, 2**31 - 1))},
        )
        self._x_current = np.asarray(self._es.ask())

    def update(self) -> None:
        fitness = np.column_stack(self._fitness) * self._weights  # [pop, objectives]
        self._es.tell(list(self._x_current), fitness.max(axis=1).tolist())
        self._x_current = np.asarray(self._es.ask())


class MoCmaOptimizer(_CmaBase):
    """
    MO-CMA-ES ((mu+lambda) with one adapted covariance and step size per parent, hypervolume selection; DEAP,
    Voss et al. 2010). The initial parents are uniform in the bounds, like the NSGA-II population.
    """

    def __init__(self, bounds: tuple[int, int], pop_size: int, num_objectives: int, solution_shape: tuple[int, ...],
                 sigma0: float = 0.2) -> None:
        from deap import base

        super().__init__(bounds, pop_size, num_objectives, solution_shape)
        self._sigma0 = sigma0 * (bounds[1] - bounds[0])
        self._strategy = None

        class _Fitness(base.Fitness):
            weights = (-1.0,) * num_objectives

        class _Individual(list):
            def __init__(self, values=()):
                super().__init__(values)
                self.fitness = _Fitness()

        self._individual = _Individual
        self._offspring = [_Individual(x) for x in np.random.uniform(*bounds, size=(pop_size, self._n_var))]
        self._x_current = np.array(self._offspring)

    def update(self) -> None:
        from deap import cma

        for individual, fitness in zip(self._offspring, np.column_stack(self._fitness)):
            individual.fitness.values = tuple(float(f) for f in fitness)
        if self._strategy is None:  # generation 0 was the parents
            self._strategy = cma.StrategyMultiObjective(
                self._offspring, sigma=self._sigma0, mu=self._pop_size, lambda_=self._pop_size)
        else:
            self._strategy.update(self._offspring)
        self._offspring = self._strategy.generate(self._individual)
        for individual in self._offspring:
            individual[:] = np.clip(individual, *self._bounds)
        self._x_current = np.array(self._offspring)

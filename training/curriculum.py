"""
Curriculum learning scheduler.

Starts training on easy scenarios (FundingCut), unlocks harder ones
(TeacherShortage, IndianContext) as training progresses.

Why curriculum matters here:
  - IndianContextScenario has ~55% random collapse rate.
  - Throwing the agent into hard scenarios from step 0 means it
    never learns a baseline policy before being overwhelmed.
  - Gradual difficulty introduction stabilizes early training.
"""

import random
import numpy as np
from typing import List, Tuple, Type
from env.scenarios.base_scenario import BaseScenario
from env.scenarios.funding_cut import FundingCutScenario
from env.scenarios.teacher_shortage import TeacherShortageScenario
from env.scenarios.indian_context import IndianContextScenario


# (min_timestep_to_unlock, scenario_class, sampling_weight)
CURRICULUM_STAGES: List[Tuple[int, Type[BaseScenario], float]] = [
    (0,        FundingCutScenario,       0.80),
    (200_000,  TeacherShortageScenario,  0.15),
    (500_000,  IndianContextScenario,    0.05),
]

# As training progresses, weights shift toward harder scenarios
LATE_STAGE_WEIGHTS: List[Tuple[int, Type[BaseScenario], float]] = [
    (0,        FundingCutScenario,       0.30),
    (200_000,  TeacherShortageScenario,  0.40),
    (500_000,  IndianContextScenario,    0.30),
]

LATE_STAGE_THRESHOLD = 700_000


def get_scenario_for_step(timestep: int, rng: random.Random = None) -> BaseScenario:
    """
    Sample a scenario appropriate for the current training timestep.
    Uses weighted random sampling from unlocked scenarios.
    """
    if rng is None:
        rng = random

    stages = LATE_STAGE_WEIGHTS if timestep >= LATE_STAGE_THRESHOLD else CURRICULUM_STAGES
    available = [(w, cls) for min_t, cls, w in stages if timestep >= min_t]

    if not available:
        return FundingCutScenario()

    total_w = sum(w for w, _ in available)
    r = rng.random() * total_w
    cumulative = 0.0
    for w, cls in available:
        cumulative += w
        if r <= cumulative:
            return cls()
    return available[-1][1]()


class CurriculumScheduler:
    """
    Stateful scheduler used by the training loop.
    Tracks timestep and returns scenarios on demand.
    """

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)
        self._timestep = 0
        self._scenario_counts = {
            "FundingCutScenario": 0,
            "TeacherShortageScenario": 0,
            "IndianContextScenario": 0,
        }

    def next_scenario(self, timestep: int) -> BaseScenario:
        self._timestep = timestep
        scenario = get_scenario_for_step(timestep, self._rng)
        key = scenario.__class__.__name__
        self._scenario_counts[key] = self._scenario_counts.get(key, 0) + 1
        return scenario

    def stats(self) -> dict:
        total = sum(self._scenario_counts.values()) or 1
        return {
            k: {"count": v, "pct": round(v / total * 100, 1)}
            for k, v in self._scenario_counts.items()
        }
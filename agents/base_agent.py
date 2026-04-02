from abc import ABC, abstractmethod
from typing import Dict
import numpy as np
from env.state import SystemState

class BaseAgent(ABC):
    def __init__(self, state: SystemState, rng: np.random.Generator):
        self.rng = rng

    @abstractmethod
    def step(self, state: SystemState, incentives: Dict[str, float]) -> None:
        """Modify system state in-place for one timestep."""
        pass

    def _clamp(self, val: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, val))
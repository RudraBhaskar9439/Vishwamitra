from abc import ABC, abstractmethod
import numpy as np
from env.state import SystemState


class BaseScenario(ABC):
    """
    Abstract base for all episode scenarios.

    A scenario controls:
    - How the system is initialized (initial_state)
    - External shocks applied mid-episode (apply_shock)
    - Scenario metadata for logging/evaluation
    """

    @abstractmethod
    def initial_state(self, rng: np.random.Generator) -> SystemState:
        """Return a freshly initialized SystemState for this scenario."""
        pass

    def apply_shock(self, state: SystemState, step: int) -> None:
        """
        Optionally modify state at a given timestep to simulate shocks.
        Override in subclasses. Default: no-op.
        """
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def difficulty(self) -> str:
        """'easy' | 'medium' | 'hard' — used by curriculum scheduler."""
        return "medium"

    def __repr__(self) -> str:
        return f"{self.name}(difficulty={self.difficulty})"
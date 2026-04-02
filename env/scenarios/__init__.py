from .base_scenario import BaseScenario
from .funding_cut import FundingCutScenario
from .teacher_shortage import TeacherShortageScenario
from .indian_context import IndianContextScenario

SCENARIO_REGISTRY = {
    "funding_cut": FundingCutScenario,
    "teacher_shortage": TeacherShortageScenario,
    "indian_context": IndianContextScenario,
}

def get_scenario(name: str) -> BaseScenario:
    if name not in SCENARIO_REGISTRY:
        raise ValueError(f"Unknown scenario '{name}'. Choose from: {list(SCENARIO_REGISTRY)}")
    return SCENARIO_REGISTRY[name]()

__all__ = [
    "BaseScenario",
    "FundingCutScenario",
    "TeacherShortageScenario",
    "IndianContextScenario",
    "SCENARIO_REGISTRY",
    "get_scenario",
]
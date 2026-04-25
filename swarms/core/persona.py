from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Persona:
    """
    A single deliberating identity. Each agent in a swarm gets exactly one
    persona; what makes agents within the same swarm different is the
    `system_prompt` field. Roles are not subclasses — a persona's `role`
    field is just a tag.
    """
    id: str
    role: str
    name: str
    system_prompt: str
    traits: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id or not self.role or not self.system_prompt:
            raise ValueError(
                f"Persona must have id, role, and system_prompt (got id={self.id!r})"
            )

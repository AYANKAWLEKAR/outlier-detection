from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from agentfw.synth.dists import IdPool
from agentfw.synth.load import load_catalog, load_personas, load_tasks, validate_world
from agentfw.synth.schema import Catalog, Persona, TaskSpec


@dataclass
class World:
    catalog: Catalog
    tasks: dict[str, TaskSpec]
    personas: dict[str, Persona]

    def persona(self, agent_id: str) -> Persona:
        return self.personas[agent_id]

    def pools(self, agent_id: str) -> dict[str, IdPool]:
        """Fresh pools for a persona, each seeded deterministically from its position."""
        p = self.personas[agent_id]
        return {
            name: IdPool(spec.prefix, spec.size, spec.s, np.random.default_rng(1000 + i))
            for i, (name, spec) in enumerate(sorted(p.id_pools.items()))
        }


@pytest.fixture(scope="session")
def world() -> World:
    catalog, tasks = load_catalog(), load_tasks()
    personas = load_personas()
    validate_world(catalog, personas, tasks)
    return World(catalog, tasks, {p.agent_id: p for p in personas})

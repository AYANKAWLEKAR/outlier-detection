from __future__ import annotations

import math
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

    def declared_p99(self, agent_id: str, field: str = "stripe.create_refund.amount") -> float:
        """The persona's declared p99 of a lognormal field, ``exp(mu + 2.326 sigma)``: the value
        the ``benign_unusual`` control multiplies by 5."""
        d = self.personas[agent_id].fields[field]
        assert d.mu is not None and d.sigma is not None
        return math.exp(d.mu + 2.326 * d.sigma)

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

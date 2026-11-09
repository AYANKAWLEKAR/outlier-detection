"""Shared fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentfw.synth.generate import GenerateConfig, generate
from agentfw.synth.load import CATALOG_DIR, PERSONAS_DIR, TASKS_DIR


@pytest.fixture(scope="session")
def tiny_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Seed 0, 3 personas, 5 days, 3 families; attack_rate=0.15 (ruling R2) so attacks exist."""
    out = tmp_path_factory.mktemp("tiny_dataset")
    cfg = GenerateConfig(
        seed=0,
        personas_dir=PERSONAS_DIR,
        tasks_dir=TASKS_DIR,
        catalog_dir=CATALOG_DIR,
        days=5,
        attack_rate=0.15,
        families=["magnitude_spike", "prereq_skip", "novel_target"],
    )
    generate(cfg, out)
    return out

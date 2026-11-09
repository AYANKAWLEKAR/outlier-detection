"""Shared fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentfw.bench.dataset import load_dir
from agentfw.core.envelope import Label, RawCall
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


@pytest.fixture(scope="session")
def validation_dataset(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[list[RawCall], dict[str, Label]]:
    """Spec 4.7 validation set: seed 0, 3 personas, 60 days, 3 families, attack_rate=0.15 (R2)."""
    out = tmp_path_factory.mktemp("validation_dataset")
    cfg = GenerateConfig(
        seed=0,
        personas_dir=PERSONAS_DIR,
        tasks_dir=TASKS_DIR,
        catalog_dir=CATALOG_DIR,
        days=60,
        attack_rate=0.15,
        families=["magnitude_spike", "prereq_skip", "novel_target"],
    )
    generate(cfg, out)
    raws, labels, _ = load_dir(out)
    return raws, labels

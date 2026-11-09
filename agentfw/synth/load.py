"""Loaders and cross-reference validation for the synthetic world."""

from __future__ import annotations

from pathlib import Path

import yaml

from agentfw.synth.schema import Catalog, Persona, TaskSpec, ToolSpec

_HERE = Path(__file__).parent
CATALOG_DIR = _HERE / "catalog"
PERSONAS_DIR = _HERE / "personas"
TASKS_DIR = _HERE / "tasks"

DERIVED_SUFFIXES = ("__domain__", "__len__", "__host__", "__prefix__")


class WorldError(ValueError):
    """Raised by ``validate_world`` with every problem found."""


def _yaml_files(d: Path) -> list[Path]:
    return sorted(d.glob("*.yaml"))


def load_catalog(directory: Path | str = CATALOG_DIR) -> Catalog:
    cat: Catalog = {}
    for f in _yaml_files(Path(directory)):
        raw = yaml.safe_load(f.read_text()) or {}
        for name, body in raw.items():
            if name in cat:
                raise WorldError(f"duplicate tool {name} in {f.name}")
            cat[name] = ToolSpec.model_validate(body)
    return cat


def load_personas(directory: Path | str = PERSONAS_DIR) -> list[Persona]:
    return [
        Persona.model_validate(yaml.safe_load(f.read_text())) for f in _yaml_files(Path(directory))
    ]


def load_tasks(directory: Path | str = TASKS_DIR) -> dict[str, TaskSpec]:
    tasks: dict[str, TaskSpec] = {}
    for f in _yaml_files(Path(directory)):
        raw = yaml.safe_load(f.read_text()) or {}
        for name, body in raw.items():
            if name in tasks:
                raise WorldError(f"duplicate task {name} in {f.name}")
            tasks[name] = TaskSpec.model_validate({"name": name, **body})
    return tasks


def _has_field(spec: ToolSpec, field: str) -> bool:
    return any(k.split(".", 1)[-1] == field for k in spec.produces)


def _check_source(source: str, catalog: Catalog) -> str | None:
    if source == "external":
        return None
    tool, _, field = source.rpartition(".")
    if tool not in catalog:
        return f"unknown tool {tool!r} in source {source!r}"
    if not _has_field(catalog[tool], field):
        return f"tool {tool} does not produce field {field!r} (source {source!r})"
    return None


def _check_catalog(catalog: Catalog, errors: list[str]) -> None:
    for name, ts in catalog.items():
        for arg, a in ts.args.items():
            if a.source is not None and (e := _check_source(a.source, catalog)):
                errors.append(f"{name}.{arg}: {e}")
            if a.type == "categorical" and not (a.vocab or (a.dist and a.dist.vocab)):
                errors.append(f"{name}.{arg}: categorical needs a vocab")
            if a.type == "numeric" and a.dist is None:
                errors.append(f"{name}.{arg}: numeric needs a dist")
            if a.vocab and a.dist and a.dist.vocab and a.vocab != a.dist.vocab:
                errors.append(f"{name}.{arg}: vocab and dist.vocab differ")
        if set(ts.produced) != set(ts.produces):
            errors.append(f"{name}: produced keys must equal produces")
        for pid in ts.response.produces_ids:
            if pid not in ts.produces:
                errors.append(f"{name}: produces_ids entry {pid!r} not in produces")


def _check_field_key(key: str, catalog: Catalog) -> str | None:
    base = key
    for suf in DERIVED_SUFFIXES:
        if key.endswith("." + suf):
            base = key[: -len(suf) - 1]
            break
    tool, _, arg = base.rpartition(".")
    if tool not in catalog:
        return f"unknown tool {tool!r} in field key {key!r}"
    if arg not in catalog[tool].args:
        return f"tool {tool} has no arg {arg!r} (field key {key!r})"
    return None


def _check_task(task: TaskSpec, catalog: Catalog, errors: list[str]) -> None:
    aliases: dict[str, str] = {}
    for i, st in enumerate(task.steps):
        where = f"task {task.name} step {i}"
        if st.tool not in catalog:
            errors.append(f"{where}: unknown tool {st.tool!r}")
            if st.as_:
                aliases[st.as_] = st.tool
            continue
        spec = catalog[st.tool]
        for arg, binding in st.args.items():
            if arg not in spec.args:
                errors.append(f"{where}: {st.tool} has no arg {arg!r}")
            if binding == "sample" or binding.startswith("ext:"):
                continue
            ref = binding
            if binding.startswith("frac:"):
                parts = binding.split(":")
                if len(parts) != 4:
                    errors.append(f"{where}: bad frac binding {binding!r}")
                    continue
                ref = parts[1]
            alias, _, field = ref.partition(".")
            if alias not in aliases or not field:
                errors.append(f"{where}: binding {binding!r} references unknown alias {alias!r}")
            elif aliases[alias] in catalog and not _has_field(catalog[aliases[alias]], field):
                errors.append(
                    f"{where}: {aliases[alias]} does not produce {field!r} (binding {binding!r})"
                )
        if st.as_:
            aliases[st.as_] = st.tool


def validate_world(catalog: Catalog, personas: list[Persona], tasks: dict[str, TaskSpec]) -> None:
    errors: list[str] = []
    _check_catalog(catalog, errors)
    for task in tasks.values():
        _check_task(task, catalog, errors)
    for p in personas:
        who = f"persona {p.agent_id}"
        for t in p.tools:
            if t not in catalog:
                errors.append(f"{who}: unknown tool {t!r}")
        for key in p.fields:
            if e := _check_field_key(key, catalog):
                errors.append(f"{who}: {e}")
        if p.drift and (e := _check_field_key(p.drift.field, catalog)):
            errors.append(f"{who}: drift {e}")
        for tw in p.tasks:
            ptask = tasks.get(tw.name)
            if ptask is None:
                errors.append(f"{who}: unknown task {tw.name!r}")
                continue
            for st in ptask.steps:
                if st.tool not in p.tools:
                    errors.append(f"{who}: task {ptask.name} uses {st.tool} not in persona tools")
                for binding in st.args.values():
                    if binding.startswith("ext:") and binding[4:] not in p.id_pools:
                        errors.append(f"{who}: task {ptask.name} needs id pool {binding[4:]!r}")
    if errors:
        raise WorldError("invalid world:\n  " + "\n  ".join(errors))

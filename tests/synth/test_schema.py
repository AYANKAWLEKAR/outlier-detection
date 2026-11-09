from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from agentfw.synth.load import (
    CATALOG_DIR,
    PERSONAS_DIR,
    TASKS_DIR,
    WorldError,
    load_catalog,
    load_personas,
    load_tasks,
    validate_world,
)
from agentfw.synth.schema import DistSpec, Persona, TaskSpec, ToolSpec

CAT, PER, TASKS = CATALOG_DIR, PERSONAS_DIR, TASKS_DIR


def test_world_loads_and_is_consistent() -> None:
    cat, ps, ts = load_catalog(CAT), load_personas(PER), load_tasks(TASKS)
    validate_world(cat, ps, ts)
    assert {p.agent_id for p in ps} == {"support-refunds-a", "sales-ops-a", "outreach-a"}
    for tool in (
        "stripe.retrieve_charge",
        "stripe.list_charges",
        "stripe.create_refund",
        "stripe.retrieve_customer",
        "stripe.create_payout",
        "salesforce.run_soql",
        "salesforce.get_record",
        "salesforce.update_record",
        "salesforce.mass_update",
        "salesforce.create_task",
        "mail.send_email",
        "slack.post_message",
        "slack.read_channel",
        "zendesk.get_ticket",
        "zendesk.list_tickets",
        "zendesk.add_comment",
        "zendesk.update_status",
    ):
        assert tool in cat


def test_spec_example_values_preserved() -> None:
    p = {p.agent_id: p for p in load_personas(PER)}["support-refunds-a"]
    amt = p.fields["stripe.create_refund.amount"]
    assert (amt.family, amt.mu, amt.sigma) == ("lognormal", 3.6, 0.7)
    assert p.session_rate.lam == 40


def test_refund_reason_and_small_refund_shape() -> None:
    cat, ts = load_catalog(CAT), load_tasks(TASKS)
    dist = cat["stripe.create_refund"].args["reason"].dist
    assert dist is not None and dist.vocab is not None and dist.weights is not None
    assert dist.weights[dist.vocab.index("manager_approved")] == 0.01
    steps = ts["small_refund"].steps
    assert [s.tool for s in steps] == [
        "zendesk.get_ticket",
        "stripe.retrieve_charge",
        "stripe.create_refund",
        "mail.send_email",
    ]
    assert steps[0].as_ == "ticket" and steps[1].skippable
    assert steps[2].args["amount"] == "sample"


def test_only_reserved_domains_in_yaml() -> None:
    import re

    import yaml

    suffixes: list[str] = []
    for d in (CAT, PER, TASKS):
        for f in Path(d).glob("*.yaml"):
            text = f.read_text()
            for dom in re.findall(r"@[\w.-]+|[\w-]+\.(?:com|org|net|io|dev|co|edu|gov)\b", text):
                assert dom.endswith((".example", ".test")), f"{dom} in {f}"
            if d == CAT:
                for body in yaml.safe_load(text).values():
                    for spec in [*body["args"].values(), *body.get("produced", {}).values()]:
                        if "domain_suffix" in spec:
                            suffixes.append(spec["domain_suffix"])
    assert suffixes and all(x in (".example", ".test") for x in suffixes)
    cat = load_catalog(CAT)
    for ts in cat.values():
        for a in [*ts.args.values(), *ts.produced.values()]:
            if a.type in ("email", "list_of_email"):
                assert a.domain_suffix in (".example", ".test")


def test_echo_len_bool_and_item_count_conventions() -> None:
    cat = load_catalog(CAT)
    assert cat["zendesk.get_ticket"].produced["ticket.id"].from_arg == "ticket_id"
    assert cat["salesforce.get_record"].produced["record.id"].from_arg == "record_id"
    pub = cat["zendesk.add_comment"].args["public"].dist
    assert pub is not None and (pub.family, pub.p) == ("bernoulli", 0.3)
    for name in (
        "zendesk.list_tickets",
        "stripe.list_charges",
        "salesforce.run_soql",
        "slack.read_channel",
    ):
        ts = cat[name]
        assert ts.response.item_count is not None
        assert all(a.len_dist is None for a in ts.produced.values())
        assert "record.count" not in ts.produced


def _world() -> tuple[dict[str, ToolSpec], list[Persona], dict[str, TaskSpec]]:
    return load_catalog(CAT), load_personas(PER), load_tasks(TASKS)


def test_unreachable_persona_override_rejected() -> None:
    cat, ps, ts = _world()
    p = {q.agent_id: q for q in ps}["support-refunds-a"]
    dist = p.fields["stripe.create_refund.amount"]
    p2 = p.model_copy(update={"fields": {**p.fields, "stripe.retrieve_charge.charge_id": dist}})
    with pytest.raises(WorldError, match=r"unreachable.*stripe\.retrieve_charge\.charge_id"):
        validate_world(cat, [p2], ts)


def test_two_domain_overrides_rejected() -> None:
    cat, ps, ts = _world()
    p = {q.agent_id: q for q in ps}["support-refunds-a"]
    d = p.fields["mail.send_email.to.__domain__"]
    p2 = p.model_copy(update={"fields": {**p.fields, "zendesk.get_ticket.ticket_id.__domain__": d}})
    with pytest.raises(WorldError, match="__domain__"):
        validate_world(cat, [p2], ts)


def test_bad_from_arg_rejected() -> None:
    cat, ps, ts = _world()
    cat["zendesk.get_ticket"].produced["ticket.id"].from_arg = "nope"
    with pytest.raises(WorldError, match="from_arg"):
        validate_world(cat, ps, ts)


def test_freetext_without_len_dist_rejected() -> None:
    cat, ps, ts = _world()
    cat["mail.send_email"].args["subject"].len_dist = None
    with pytest.raises(WorldError, match=r"mail\.send_email\.subject.*len_dist"):
        validate_world(cat, ps, ts)


def test_bool_without_bernoulli_rejected() -> None:
    cat, ps, ts = _world()
    cat["zendesk.add_comment"].args["public"].dist = None
    with pytest.raises(WorldError, match="bernoulli"):
        validate_world(cat, ps, ts)


def test_produced_list_with_own_len_dist_on_item_count_tool_rejected() -> None:
    cat, ps, ts = _world()
    cat["zendesk.list_tickets"].produced["ticket.ids"].len_dist = DistSpec(family="poisson", lam=3)
    with pytest.raises(WorldError, match=r"ticket\.ids.*item_count"):
        validate_world(cat, ps, ts)


def test_validate_world_reports_unknown_tool_and_bad_source(tmp_path: Path) -> None:
    cat_dir = tmp_path / "catalog"
    shutil.copytree(CAT, cat_dir)
    (cat_dir / "bad.yaml").write_text(
        "stripe.bad:\n  kind: read\n  args:\n"
        "    x: {type: id, prefix: 'x_', source: stripe.retrieve_charge.missing}\n"
    )
    (tmp_path / "p.yaml").write_text(
        "agent_id: bad\ntools: [stripe.nope]\nsession_rate: {family: poisson, lam: 1}\n"
        "tasks: []\nfields: {stripe.nope.amount: {family: constant, value: 1}}\n"
    )
    cat, ps = load_catalog(cat_dir), [load_personas(tmp_path)[0]]
    with pytest.raises(WorldError, match=r"stripe\.nope") as ei:
        validate_world(cat, ps, load_tasks(TASKS))
    assert "missing" in str(ei.value)


def test_derived_suffix_keys_accepted_and_bad_arg_rejected() -> None:
    cat, ts = load_catalog(CAT), load_tasks(TASKS)
    p = {q.agent_id: q for q in load_personas(PER)}["support-refunds-a"]
    for suf in ("__domain__", "__len__"):
        p2 = p.model_copy(
            update={
                "fields": {f"mail.send_email.to.{suf}": p.fields["stripe.create_refund.amount"]}
            }
        )
        validate_world(cat, [p2], ts)
    p4 = p.model_copy(
        update={"fields": {"mail.send_email.to.__host__": p.fields["stripe.create_refund.amount"]}}
    )
    with pytest.raises(
        WorldError, match="unreachable"
    ):  # known derived suffix, but nothing consumes it
        validate_world(cat, [p4], ts)
    p3 = p.model_copy(
        update={"fields": {"mail.send_email.zzz": p.fields["stripe.create_refund.amount"]}}
    )
    with pytest.raises(WorldError, match="zzz"):
        validate_world(cat, [p3], ts)

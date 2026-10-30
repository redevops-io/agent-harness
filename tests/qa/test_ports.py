# SPDX-License-Identifier: AGPL-3.0-or-later
"""Neutral ports: structural conformance and parity with the frozen protocol inventory."""
import dataclasses
import inspect
import re
import typing

import pytest

from agent_harness.qa import ports, records as R, schemas

from ._helpers import load_inventory

INV = load_inventory()

PAIRS = [
    ("QAAdapter", ports.NeutralQAAdapter),
    ("CheckAdapter", ports.NeutralCheckAdapter),
    ("Redactor", ports.Redactor),
    ("BlobStore", ports.BlobSink),
]


def _methods(proto):
    return {n: f for n, f in vars(proto).items() if inspect.isfunction(f) and not n.startswith("_")}


@pytest.mark.parametrize("frozen,mine", PAIRS)
def test_method_and_parameter_names_equal_the_frozen_protocol(frozen, mine):
    spec = INV["protocols"][frozen]
    methods = _methods(mine)
    assert set(methods) == set(spec["methods"])
    for name, body in spec["methods"].items():
        sig = inspect.signature(methods[name])
        assert [p for p in sig.parameters if p != "self"] == body["parameters"], (frozen, name)
        assert str(sig.return_annotation) == body["returns"], (frozen, name)
    attrs = set(typing.get_type_hints(mine)) if spec["attributes"] else set()
    assert attrs == set(spec["attributes"])


def test_api_versions_are_the_frozen_strings():
    assert schemas.ADAPTER_API_VERSION == INV["api_versions"]["QAAdapter"] == "qa-adapter/1"
    assert schemas.CHECK_ADAPTER_API_VERSION == INV["api_versions"]["CheckAdapter"] == "qa-check-adapter/1"
    assert schemas.NEUTRAL_API == "qa-neutral/1" and schemas.CONTRACT_ID == INV["contract"] == "qa/v1"
    assert schemas.HASH_ALGORITHM == "canon-v1" and schemas.CONTRACT_HASH_ALGORITHM == "rcv1"


def test_the_port_list_is_the_adr_list():
    assert [p.__name__ for p in ports.PORTS] == ["NeutralQAAdapter", "NeutralCheckAdapter", "Redactor", "BlobSink",
                                                  "Recoverer", "ActionPolicy", "Clock"]


def test_adapters_never_receive_an_expected_value_nor_return_a_verdict():
    forbidden_params = {"expected", "verdict", "predicate"}
    forbidden_returns = re.compile(r"\b(Verdict|AssertionResult|Check|Outcome|bool)\b")
    for proto in (ports.NeutralQAAdapter, ports.NeutralCheckAdapter):
        for name, fn in _methods(proto).items():
            sig = inspect.signature(fn)
            assert forbidden_params.isdisjoint(sig.parameters), (proto.__name__, name)
            assert not forbidden_returns.search(str(sig.return_annotation)), (proto.__name__, name)
    assert str(inspect.signature(ports.NeutralQAAdapter.assert_).return_annotation) == "ObservedValue"
    assert str(inspect.signature(ports.NeutralCheckAdapter.run_check).return_annotation) == "ObservedValue"
    assert list(inspect.signature(ports.NeutralQAAdapter.act).parameters) == ["self", "s", "action", "target"]
    assert inspect.signature(ports.NeutralQAAdapter.locate).parameters["mode"].kind is inspect.Parameter.KEYWORD_ONLY


class FakeAdapter:
    adapter_id = "fake"
    adapter_version = "0"
    api_version = "qa-adapter/1"

    def open_session(self, profile): ...
    def observe(self, s, req): ...
    def locate(self, s, target, *, mode): ...
    def act(self, s, action, target): ...
    def assert_(self, s, subject): ...
    def capture(self, s, kinds): ...
    def reset(self, s, scope): ...
    def close_session(self, s): ...


class FakeChecks:
    adapter_id = "fake"
    adapter_version = "0"
    api_version = "qa-check-adapter/1"

    def open_workspace(self, profile): ...
    def resolve(self, w, ref): ...
    def run_check(self, w, subject): ...
    def capture(self, w, kinds): ...
    def reset(self, w, scope): ...
    def close_workspace(self, w): ...


def test_a_structurally_matching_object_satisfies_the_ports_without_importing_them():
    assert isinstance(FakeAdapter(), ports.NeutralQAAdapter)
    assert isinstance(FakeChecks(), ports.NeutralCheckAdapter)
    assert not isinstance(FakeAdapter(), ports.NeutralCheckAdapter)
    assert not isinstance(FakeChecks(), ports.NeutralQAAdapter)

    class Combined(FakeAdapter, FakeChecks):          # one adapter can satisfy both forms (the bridge's shim)
        pass

    assert isinstance(Combined(), ports.NeutralQAAdapter) and isinstance(Combined(), ports.NeutralCheckAdapter)


def test_a_missing_method_or_attribute_is_not_conformance():
    class NoAssert(FakeAdapter):
        assert_ = None

    class NoVersion:
        pass

    for name in ("assert_", "locate", "act", "close_session"):
        broken = type("Broken", (FakeAdapter,), {name: None})()
        assert not isinstance(broken, ports.NeutralQAAdapter), name
    assert not isinstance(NoVersion(), ports.NeutralQAAdapter)
    attr_less = type("AttrLess", (), {n: getattr(FakeAdapter, n) for n in _methods(ports.NeutralQAAdapter)})()
    assert not isinstance(attr_less, ports.NeutralQAAdapter)


def test_the_small_ports_are_structural_too():
    class Red:
        policy_id = "sha256:" + "0" * 64

        def redact(self, data, media_type):
            return data

    class Blobs:
        def put(self, data, media_type): ...
        def get(self, digest): ...
        def exists(self, digest): ...
        def size(self, digest): ...

    class Rec:
        def propose(self, ctx):
            return []

    class Pol:
        def decide(self, run_id, step, attempt): ...

    class Clk:
        def now_utc(self):
            return "2026-10-30T00:00:00Z"

        def monotonic_ms(self):
            return 0

    assert isinstance(Red(), ports.Redactor) and isinstance(Blobs(), ports.BlobSink)
    assert isinstance(Rec(), ports.Recoverer) and isinstance(Pol(), ports.ActionPolicy) and isinstance(Clk(), ports.Clock)
    assert not isinstance(Rec(), ports.ActionPolicy) and not isinstance(object(), ports.Clock)


def test_recovery_context_has_no_assertion_field_and_is_frozen():
    fields = {f.name: f for f in dataclasses.fields(ports.RecoveryContext)}
    assert set(fields) == {"step", "observation"}
    hints = typing.get_type_hints(ports.RecoveryContext)
    assert hints == {"step": R.Step, "observation": R.Observation}
    banned = {R.AssertionSpec, R.AssertionSubject, R.Predicate, R.AssertionResult, R.RunManifest, R.TestSpec}
    assert banned.isdisjoint(hints.values())
    # nothing reachable from the step or the observation is an assertion either
    seen, todo = set(), [R.Step, R.Observation]
    while todo:
        cls = todo.pop()
        if cls in seen:
            continue
        seen.add(cls)
        for info in R.field_infos(cls):
            spec = info.spec.elem if info.spec.kind == "tuple" else info.spec
            if spec.kind == "record":
                todo.append(spec.cls)
    assert banned.isdisjoint(seen), banned & seen
    ctx = ports.RecoveryContext(step=None, observation=None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.step = 1
    with pytest.raises(TypeError):
        ports.RecoveryContext(step=None, observation=None, expected="x")


def test_the_only_recoverer_output_type_is_the_strict_candidate():
    sig = inspect.signature(ports.Recoverer.propose)
    assert str(sig.return_annotation) == "Sequence[LocatorCandidate]"
    assert R.LocatorCandidate.ALLOW_EXTRAS is False

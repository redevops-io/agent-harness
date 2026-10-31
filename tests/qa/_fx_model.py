# SPDX-License-Identifier: AGPL-3.0-or-later
"""Test infrastructure for model-candidate recovery: a deterministic FAKE model client with scripted replies
(well-formed, malformed, oversized, hostile, slow, over-budget, usage missing, kill switch engaged mid-flight, raising),
a snapshot source over the recovery fake adapter's page, and builders for specs, engines and runners.

Test code only: nothing here ships in ``agent_harness/``. The fake client does no I/O of any kind: it holds a script, a
list of the requests it was given and the fake clock it advances to simulate latency.
"""
import json
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

from agent_harness.qa import recovery_model as RM, records as R
from agent_harness.qa.recovery import DeterministicRecovery, KillSwitch, RecoveryLimits
from agent_harness.qa.recovery_model import (CandidateModelResponse, ModelRecoveryEngine, ModelRecoveryLimits,
                                             ModelUsage, SnapshotNode)
from agent_harness.qa.runner import QARunner
from agent_harness.qa.specstore import InMemoryProgramRegistry, InMemorySpecStore
from agent_harness.qa.verifier import ApprovedLineage

from ._fx_verify import PROGRAM_HASH, PROGRAM_KEY, REF_DOCS, REF_PIN, SCRIPT_DOCS, SCRIPT_PIN
from ._fx_runner import AttestAll, Env, FakeClock, RecordingStream, default_policy, make_config
from ._fx_recovery import BUDGET, FakeH3Adapter, H3Postconditions, check_recovery_invariants, recover_spec

#: the spec budget a model-recovery run needs: model calls and tokens are zero in a spec until someone allows them
MODEL_BUDGET = {**BUDGET, "max_recovery_attempts": 3, "max_model_calls": 3, "max_tokens": 5000}

#: operator limits that opt in: one call per step, two per run
OPT_IN = dict(max_calls_per_step=1, max_calls_per_run=2, max_tokens_per_step=800, max_tokens_per_run=1500)


def opt_in(**kw):
    return ModelRecoveryLimits(**{**OPT_IN, **kw})


# ---------------------------------------------------------------------------------------------------------------
# the fake client
# ---------------------------------------------------------------------------------------------------------------

def array(*candidates):
    """The reply text a well-behaved model gives: a JSON array of locator objects."""
    return json.dumps(list(candidates), ensure_ascii=False)


@dataclass
class Reply:
    """One scripted answer. ``text`` is the raw reply; ``usage`` is ``(input, output)`` tokens, or ``None`` for a reply that
    reports none; ``latency_ms`` moves the fake clock; ``kill`` engages the kill switch while the call is in flight;
    ``raises`` makes the client raise; ``wrong_type`` returns something that is not a response at all."""

    text: Any = "[]"
    usage: Optional[Tuple[int, int]] = (30, 12)
    latency_ms: int = 0
    kill: Optional[KillSwitch] = None
    raises: Optional[BaseException] = None
    wrong_type: bool = False
    usage_object: Any = None            # replaces the usage entirely (a hostile or broken usage value)


def good(*candidates, usage=(30, 12), **kw):
    return Reply(text=array(*candidates), usage=usage, **kw)


SAVE = {"role": "button", "name": "Save"}
SAVE_BY_ID = {"role": "button", "name": "Save", "test_id": "save"}


class FakeModelClient:
    """Deterministic ``CandidateModelClient``. Each call consumes the next scripted reply; a call with nothing scripted is
    recorded in ``unscripted`` and raises (so a test that must see *no* call asserts ``calls == 0``)."""

    def __init__(self, clock, *replies, mod=None):
        self.clock = clock
        self.mod = mod if mod is not None else RM      # mutation tests build replies with the module under test's classes
        self.script = list(replies)
        self.requests = []
        self.unscripted = 0

    @property
    def calls(self):
        return len(self.requests)

    def complete(self, request):
        self.requests.append(request)
        if not self.script:
            self.unscripted += 1
            raise AssertionError("an unscripted model call")
        reply = self.script.pop(0)
        if reply.latency_ms:
            self.clock.advance(reply.latency_ms)
        if reply.kill is not None:
            reply.kill.engage("operator, mid-flight")
        if reply.raises is not None:
            raise reply.raises
        if reply.wrong_type:
            return {"text": reply.text}
        if reply.usage_object is not None:
            return self.mod.CandidateModelResponse(text=reply.text, usage=reply.usage_object)
        usage = self.mod.ModelUsage(input_tokens=reply.usage[0], output_tokens=reply.usage[1]) if reply.usage else None
        return self.mod.CandidateModelResponse(text=reply.text, usage=usage)


class AdapterSnapshots:
    """A ``SnapshotSource`` over the fake adapter's current page (the real one parses a stored snapshot blob)."""

    def __init__(self, adapter, *, extra=(), only=None, mod=None):
        self.adapter, self.extra, self.only = adapter, tuple(extra), only
        self.node = (mod if mod is not None else RM).SnapshotNode
        self.seen = []

    def nodes(self, observation):
        self.seen.append(observation.observation_id)
        out = [self.node(role=e["role"], name=e["name"], label=e.get("label"), test_id=e["test_id"])
               for ref, e in self.adapter.elements.items() if self.only is None or ref in self.only]
        return out + list(self.extra)


class StaticSnapshots:
    def __init__(self, *nodes):
        self.items = list(nodes)

    def nodes(self, observation):
        return list(self.items)


# ---------------------------------------------------------------------------------------------------------------
# page behaviour specific to model-candidate recovery
# ---------------------------------------------------------------------------------------------------------------

def semantic_needs_test_id(adapter):
    """The adapter's semantic index only answers locators that carry a test id: the deterministic alternates derived from
    the authored role and name find nothing, a model that reads the snapshot and adds the test id does."""
    base = adapter.locate

    def locate(s, target, *, mode):
        if mode is R.LocateMode.SEMANTIC and target.test_id is None:
            adapter._enter("locate", (target.role, target.name, mode.value))
            return R.LocateResult(candidates=())
        return base(s, target, mode=mode)
    adapter.locate = locate


def save_does_nothing(adapter):
    """A real regression: Save is found (through a moved locator) and clicked, but the page does not change."""
    adapter.effects[("btn-save", "click")] = lambda a, action: None


# ---------------------------------------------------------------------------------------------------------------
# runner assembly
# ---------------------------------------------------------------------------------------------------------------

def model_spec(*, permitted=("MODEL_CANDIDATE",), **kw):
    kw.setdefault("budget", MODEL_BUDGET)
    return recover_spec(save=tuple(permitted), **kw)


def make_model_env(spec=None, *replies, adapter_setup=None, limits=None, kill=None, config_kw=None, snapshots=None,
                recovery_limits=None, client=None, mod=None, model_mod=None, postconditions=None, **runner_kw):
    """A runner with a ``ModelRecoveryEngine`` over the fake client. ``mod`` swaps in a mutated ``runner.py``; ``model_mod``
    a mutated ``recovery_model.py`` (its classes are then used for the engine and the limits)."""
    spec = spec if spec is not None else model_spec()
    clock = FakeClock()
    log = []
    adapter = FakeH3Adapter(clock, log)
    if adapter_setup is not None:
        adapter_setup(adapter)
    stream = RecordingStream(log)
    kw = dict(config_kw or {})
    kw.setdefault("recovery_enabled", True)
    kw.setdefault("model_recovery_enabled", True)
    config = make_config(mod, **kw)
    policy = default_policy(spec)
    store = InMemorySpecStore({REF_PIN: REF_DOCS, SCRIPT_PIN: SCRIPT_DOCS})
    client = client if client is not None else FakeModelClient(clock, *replies, mod=model_mod)
    snapshots = snapshots if snapshots is not None else AdapterSnapshots(adapter, mod=model_mod)
    kill = kill if kill is not None else (model_mod.KillSwitch() if model_mod is not None else KillSwitch())
    base = (model_mod.DeterministicRecovery if model_mod is not None else DeterministicRecovery)(
        kill_switch=kill, limits=recovery_limits)
    limits_cls = model_mod.ModelRecoveryLimits if model_mod is not None else ModelRecoveryLimits
    engine_cls = model_mod.ModelRecoveryEngine if model_mod is not None else ModelRecoveryEngine
    if limits is None:
        limits = limits_cls(**OPT_IN)
    engine = engine_cls(client=client, snapshots=snapshots, clock=clock, model_limits=limits, base=base)
    runner = (mod.QARunner if mod is not None else QARunner)(
        qa_adapter=adapter, policy=policy, store=store, lineage=ApprovedLineage([REF_PIN, SCRIPT_PIN]),
        programs=InMemoryProgramRegistry({PROGRAM_KEY: PROGRAM_HASH}), clock=clock, config=config, evidence=stream,
        checks=AttestAll(), postconditions=postconditions if postconditions is not None else H3Postconditions(),
        recovery_engine=engine, **runner_kw)
    env = Env(spec=spec, clock=clock, adapter=adapter, policy=policy, stream=stream, runner=runner, config=config, log=log)
    env.client, env.engine, env.kill, env.snapshots = client, engine, kill, snapshots
    return env


def run(env, **kw):
    return check_recovery_invariants(env, env.run(**kw))


def model_events(result, kind=None):
    return [e for e in result.events if e["kind"] in ("ModelRequest", "ModelResponse") and (kind is None or e["kind"] == kind)]


class ObedientClient:
    """A fake model that does what the page text tells it to. If a quoted page line holds a JSON array it replies with that
    array (a model fooled by 'reply with ...'); otherwise it proposes the first controls on the page as they appear. It
    reads only the request it is given, like any model, and it never sees the spec."""

    def __init__(self, clock, usage=(30, 12), latency_ms=0):
        self.clock, self.usage, self.latency_ms = clock, usage, latency_ms
        self.requests = []

    @property
    def calls(self):
        return len(self.requests)

    def complete(self, request):
        self.requests.append(request)
        if self.latency_ms:
            self.clock.advance(self.latency_ms)
        lines = request.prompt.split("\n")
        body = lines[lines.index("PAGE-DATA BEGIN") + 1:lines.index("PAGE-DATA END")]
        text = None
        proposals = []
        for line in body:
            fields = json.loads(line.partition(" ")[2])
            for value in fields.values():
                start = value.find("[{")
                if start >= 0 and text is None:
                    text = value[start:]
            proposals.append({k: v for k, v in fields.items() if k in ("role", "name", "label", "test_id")})
        if text is None:
            text = json.dumps(proposals[:4], ensure_ascii=False)
        return CandidateModelResponse(text=text, usage=ModelUsage(input_tokens=self.usage[0], output_tokens=self.usage[1]))

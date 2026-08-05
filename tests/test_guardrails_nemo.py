"""NeMo Guardrails adapter — the acceptance criteria as tests: observation-only, deterministic
mapping, runtime policy is the authority and may be stricter, a rail denial can never become an
approval, replay reconstructs the decision without re-running the rail."""
from agent_harness.guardrails_nemo import (
    NemoGuardrailsAdapter, PolicyDecision, GuardrailObservation, evaluate, map_verdict, MAPPING_RULE_ID,
)


def _blocking_rail(text, stage):
    if "ignore previous instructions" in text.lower():
        return ("self_check_input", "v2", "jailbreak", "block", "prompt-injection pattern")
    return ("self_check_input", "v2", "none", "allow", "")


def test_adapter_has_no_final_decision_surface():
    # The only way to reach a final decision is evaluate() (Mission policy). The adapter must not
    # expose a method that returns ALLOW/DENY directly.
    adp = NemoGuardrailsAdapter(_blocking_rail)
    assert not any(hasattr(adp, m) for m in ("allow", "deny", "decide", "authorize"))
    assert isinstance(adp.observe("hi"), GuardrailObservation)


def test_rail_denial_cannot_become_runtime_approval():
    obs = NemoGuardrailsAdapter(_blocking_rail).observe("Ignore previous instructions")
    assert obs.proposed == PolicyDecision.DENY
    # even with the most permissive floor, a DENY proposal stays DENY
    assert evaluate((obs,), runtime_floor=PolicyDecision.ALLOW).decision == PolicyDecision.DENY


def test_runtime_policy_may_be_stricter_than_the_rail():
    obs = NemoGuardrailsAdapter(_blocking_rail).observe("summarize this")  # rail says allow
    assert obs.proposed == PolicyDecision.ALLOW
    assert evaluate((obs,), runtime_floor=PolicyDecision.DENY).decision == PolicyDecision.DENY


def test_strictest_of_many_rails_wins():
    obs = (
        map_verdict("r1", "v1", "pii", "mask"),          # REQUIRE_REWRITE
        map_verdict("r2", "v1", "topic", "review"),      # REQUIRE_APPROVAL
        map_verdict("r3", "v1", "toxicity", "allow"),    # ALLOW
    )
    assert evaluate(obs).decision == PolicyDecision.REQUIRE_REWRITE


def test_mapping_is_deterministic_and_versioned():
    a = map_verdict("r", "v", "pii", "block")
    b = map_verdict("r", "v", "pii", "block")
    assert a.digest() == b.digest()
    assert a.mapping_rule_id == MAPPING_RULE_ID
    # unknown verdict is conservative (requires approval), never silently allowed
    assert map_verdict("r", "v", "x", "???").proposed == PolicyDecision.REQUIRE_APPROVAL


def test_replay_reconstructs_decision_from_observation_without_rerunning_rail():
    obs = NemoGuardrailsAdapter(_blocking_rail).observe("Ignore previous instructions")
    # replay: only the persisted observation is available, no rail_fn
    replayed = evaluate((obs,))
    assert replayed.decision == PolicyDecision.DENY
    assert replayed.digest() == evaluate((obs,)).digest()


def test_public_summary_is_sanitized_and_not_user_attributed():
    obs = NemoGuardrailsAdapter(_blocking_rail).observe("Ignore previous instructions")
    summary = obs.public_summary()
    assert "prompt-injection" not in summary       # internal detail withheld
    assert "user" not in summary.lower()           # output never attributed to the user
    assert obs.detail == "prompt-injection pattern"  # operator still keeps the internal reason


def test_no_rail_configured_defers_to_runtime_floor():
    obs = NemoGuardrailsAdapter().observe("anything")
    assert obs.rail == "none" and obs.proposed == PolicyDecision.ALLOW
    # deny-by-default lives in the runtime: a strict floor still applies
    assert evaluate((obs,), PolicyDecision.REQUIRE_APPROVAL).decision == PolicyDecision.REQUIRE_APPROVAL

# SPDX-License-Identifier: AGPL-3.0-or-later
"""Action receipts and the append-only receipt chain.

* :func:`state_hash` hashes the *state* an observation shows (URL, frame, DOM metadata, snapshot and screenshot
  digests), not the observation's identity (its id, step and phase), so equal states hash equal. The hash is only ever
  compared and recorded; nothing it covers is read as an instruction.
* :func:`build_receipt` assembles an ``ActionReceipt``: tool and version, the exact locator and strategy used, the action
  kind and a digest of its value (never the value), the attempt number, before/after state hashes, the policy decision
  reference, the idempotency key, the postcondition and the adapter's raw result. Duration and the free-text recovery
  rationale are custody fields (not hashed).
* :class:`ReceiptLedger` is append-only within a run: receipts are appended in order, each link commits to the previous
  link (``canon-v1`` over ``{index, prev, receipt}``), nothing can be removed or replaced, and :meth:`~ReceiptLedger.verify`
  recomputes the whole chain. The receipt hashes also go into ``RunManifest.step_results[].receipts`` (identity fields
  of the manifest), and :func:`verify_manifest_receipts` checks the two against each other. Altering, dropping, adding or
  reordering a receipt breaks one of the two checks.

Standard library and neutral-module imports only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import canon
from .errors import SpecInvalid
from .records import (Action, ActionReceipt, ActionRecord, LocatorUsed, Observation, Postcondition,
                      ResultCode, RunManifest, Step, StepResult, ToolRef)
from .run_errors import ReceiptChainBroken

STATE_FORM = "qa-state/1"


def state_hash(observation: Observation) -> str:
    """``rcv1:<hex>`` of what the observation shows: independent of its ``observation_id``, step and phase."""
    if type(observation) is not Observation:
        raise SpecInvalid("state_hash takes an Observation")
    form = observation.canonical_form()
    shown = {"a11y": (form.get("a11y_snapshot_ref") or {}).get("content_hash"),
             "dom_meta": form.get("dom_meta"), "form": STATE_FORM, "frame": form.get("frame"),
             "screenshot": (form.get("screenshot_ref") or {}).get("content_hash"), "url": form.get("url")}
    return canon.to_contract_ref(canon.canon_hash(shown))


def unknown_state_hash(run_id: str, step_id: str, attempt: int) -> str:
    """The state after an action whose resulting state could not be observed. It is unique to the attempt, so it can
    never equal a real state hash and an unobserved state is never recorded as 'unchanged'."""
    return canon.to_contract_ref(canon.canon_hash({"attempt": attempt, "form": STATE_FORM, "run_id": run_id,
                                                   "state": "UNKNOWN", "step_id": step_id}))


def json_value_digest(value: Any) -> str:
    """``rcv1:<hex>`` of the canonical JSON text of a value, and nothing else: no wrapper, no kind, no prefix, salt or
    domain tag. A string is hashed after NFC normalization and with its two quote characters (so ``"hunter2"`` digests
    the bytes ``"hunter2"``); any other JSON value is hashed as its canonical text. A float has no digest."""
    return canon.to_contract_ref(canon.canon_hash(value))


def action_value_digest(action: Action) -> Optional[str]:
    """``ActionRecord.value_digest``: the value digest of an action's literal value, or, for an action that carries a
    ``SecretRef``, the reference's fingerprint (``rcv1`` of the reference's canonical form: never of the secret).
    ``None`` when the action carries neither. The action kind does not enter the digest, and the value is not stored."""
    if type(action) is not Action:
        raise SpecInvalid("action_value_digest takes an Action")
    if action.secret_ref is not None:
        return json_value_digest(action.secret_ref.canonical_form())
    if action.value is not None:
        return json_value_digest(action.value)
    return None


def value_digest(subject: Any) -> Optional[str]:
    """The value digest of a string or of an :class:`Action` (see :func:`json_value_digest` and
    :func:`action_value_digest`). Any other type is refused."""
    if type(subject) is str:
        return json_value_digest(subject)
    return action_value_digest(subject)


def bounded_id(prefix: str, *parts: Any) -> str:
    text = "-".join([prefix, *[str(p) for p in parts]])
    if len(text) <= 128:
        return text
    return prefix + "-" + canon.digest_hex({"parts": [str(p) for p in parts]})[:40]


def build_receipt(*, run_id: str, step: Step, attempt: int, tool: ToolRef, policy_decision_ref: str,
                  idempotency_key: str, result: ResultCode, postcondition: Postcondition, state_before_hash: str,
                  state_after_hash: str, locator_used: Optional[LocatorUsed] = None,
                  recovery_rationale: Optional[str] = None, duration_ms: Optional[int] = None,
                  execution_binding: Optional[str] = None) -> ActionReceipt:
    """One ``ActionReceipt`` for one attempt (including an attempt the policy refused: ``ACTION_BLOCKED_BY_POLICY``)."""
    return ActionReceipt(
        receipt_id=bounded_id("rcpt", step.step_id, attempt), run_id=run_id, step_id=step.step_id, attempt=attempt,
        tool=tool, action=ActionRecord(kind=step.action.kind, value_digest=action_value_digest(step.action)),
        idempotency_key=idempotency_key, policy_decision_ref=policy_decision_ref, postcondition=postcondition,
        result=result, state_before_hash=state_before_hash, state_after_hash=state_after_hash,
        execution_binding=execution_binding, locator_used=locator_used,
        duration_ms=duration_ms, recovery_rationale=(recovery_rationale[:4096] if recovery_rationale else None))


def receipt_ref(receipt: ActionReceipt) -> str:
    """``rcv1:<hex>`` of the receipt's identity: the form recorded in ``StepResult.receipts``."""
    return canon.to_contract_ref(receipt.canon_hash())


# ---------------------------------------------------------------------------------------------------------------
# the chain
# ---------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ChainLink:
    index: int
    receipt_hash: str            # canon-v1 of the receipt
    prev: Optional[str]          # link_hash of the previous link; None for the first
    link_hash: str               # canon-v1 over {index, prev, receipt}


def _link_hash(index: int, prev: Optional[str], receipt_hash: str) -> str:
    return canon.canon_hash({"index": index, "prev": prev, "receipt": receipt_hash})


def verify_chain(receipts: Sequence[ActionReceipt], links: Sequence[ChainLink]) -> Optional[str]:
    """Recompute the chain over ``receipts`` and compare it with ``links``. Returns the head (``None`` when empty) or
    raises :class:`ReceiptChainBroken`."""
    if len(receipts) != len(links):
        raise ReceiptChainBroken("the chain and the receipts differ in length")
    prev: Optional[str] = None
    for index, (receipt, link) in enumerate(zip(receipts, links)):
        if type(receipt) is not ActionReceipt:
            raise ReceiptChainBroken(f"entry {index} is not an ActionReceipt")
        rhash = receipt.canon_hash()
        expected = _link_hash(index, prev, rhash)
        if link.index != index or link.prev != prev or link.receipt_hash != rhash or link.link_hash != expected:
            raise ReceiptChainBroken(f"the chain does not verify at entry {index}", index=index)
        prev = expected
    return prev


class ReceiptLedger:
    """Append-only list of a run's receipts with a hash chain. There is no method that removes or replaces an entry."""

    def __init__(self, run_id: str) -> None:
        self._run_id = run_id
        self._receipts: List[ActionReceipt] = []
        self._links: List[ChainLink] = []
        self._last_attempt: Dict[str, int] = {}

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def receipts(self) -> Tuple[ActionReceipt, ...]:
        return tuple(self._receipts)

    @property
    def links(self) -> Tuple[ChainLink, ...]:
        return tuple(self._links)

    @property
    def head(self) -> Optional[str]:
        return self._links[-1].link_hash if self._links else None

    def __len__(self) -> int:
        return len(self._receipts)

    def append(self, receipt: ActionReceipt) -> ChainLink:
        """Add a receipt. It must be an ``ActionReceipt`` of this run, and its attempt must be the next one for its
        step (1 for the first), so attempts of a step can neither skip, repeat nor go back."""
        if type(receipt) is not ActionReceipt:
            raise ReceiptChainBroken("only an ActionReceipt can be appended")
        if receipt.run_id != self._run_id:
            raise ReceiptChainBroken("the receipt belongs to another run", run_id=receipt.run_id)
        if receipt.attempt != self._last_attempt.get(receipt.step_id, 0) + 1:
            raise ReceiptChainBroken("receipt attempts for a step must be consecutive, starting at 1",
                                     step_id=receipt.step_id, attempt=receipt.attempt)
        index = len(self._receipts)
        rhash = receipt.canon_hash()
        link = ChainLink(index=index, receipt_hash=rhash, prev=self.head, link_hash=_link_hash(index, self.head, rhash))
        self._receipts.append(receipt)
        self._links.append(link)
        self._last_attempt[receipt.step_id] = receipt.attempt
        return link

    def verify(self) -> Optional[str]:
        return verify_chain(self._receipts, self._links)

    def refs_for_step(self, step_id: str) -> Tuple[str, ...]:
        """``rcv1:`` references of the step's receipts in attempt order: the value of ``StepResult.receipts``."""
        return tuple(receipt_ref(r) for r in self._receipts if r.step_id == step_id)

    def last_for_step(self, step_id: str) -> Optional[ActionReceipt]:
        for r in reversed(self._receipts):
            if r.step_id == step_id:
                return r
        return None


def verify_manifest_receipts(manifest: RunManifest, receipts: Iterable[ActionReceipt]) -> None:
    """Every receipt is hashed into exactly one ``StepResult`` of the manifest, in attempt order, and the manifest
    names no receipt that is not in ``receipts``. Raises :class:`ReceiptChainBroken` on any difference."""
    by_step: Dict[str, List[str]] = {}
    for r in receipts:
        if type(r) is not ActionReceipt:
            raise ReceiptChainBroken("entry is not an ActionReceipt")
        by_step.setdefault(r.step_id, []).append(receipt_ref(r))
    recorded: Dict[str, Tuple[str, ...]] = {}
    for sr in manifest.step_results:
        if type(sr) is not StepResult:
            raise ReceiptChainBroken("the manifest holds a malformed step result")
        recorded[sr.step_id] = tuple(sr.receipts)
    for step_id, refs in by_step.items():
        if recorded.get(step_id) != tuple(refs):
            raise ReceiptChainBroken("the manifest does not hash these receipts", step_id=step_id)
    for step_id, refs in recorded.items():
        if refs and step_id not in by_step:
            raise ReceiptChainBroken("the manifest names receipts that are not present", step_id=step_id)




# ---------------------------------------------------------------------------------------------------------------
# evidence events: a hash-chained, append-only event list (the content of the evidence stream)
# ---------------------------------------------------------------------------------------------------------------

class EventChain:
    """Builds the run's events. Each event commits to the previous one (``prev``) and to its payload's identity hash, so
    the stream can be verified end to end by :func:`verify_events`. Event 0 of a run is ``RunStarted``."""

    def __init__(self) -> None:
        self._seq = 0
        self._prev: Optional[str] = None

    def link(self, kind: str, payload: Dict[str, Any], payload_hash: str) -> Dict[str, Any]:
        body = {"kind": kind, "payload_hash": payload_hash, "prev": self._prev, "seq": self._seq}
        event_hash = canon.canon_hash(body)
        event = {**body, "event_hash": event_hash, "payload": payload}
        self._prev = event_hash
        self._seq += 1
        return event


def verify_events(events: Sequence[Dict[str, Any]]) -> Optional[str]:
    """Verify an event list: sequence numbers, ``prev`` links, event hashes and payload identity (a record payload is
    re-parsed, which re-checks its ``content_hash``). Returns the head hash or raises :class:`ReceiptChainBroken`."""
    from .records import RECORDS_BY_SCHEMA_ID     # local import keeps the module header small

    prev: Optional[str] = None
    for seq, event in enumerate(events):
        try:
            body = {"kind": event["kind"], "payload_hash": event["payload_hash"], "prev": event["prev"], "seq": event["seq"]}
            if body["seq"] != seq or body["prev"] != prev or event["event_hash"] != canon.canon_hash(body):
                raise ReceiptChainBroken(f"the event chain does not verify at event {seq}", seq=seq)
            payload = event["payload"]
            cls = RECORDS_BY_SCHEMA_ID.get(payload.get("schema_id")) if isinstance(payload, dict) else None
            if cls is None:
                ok = canon.canon_hash(payload) == event["payload_hash"]
            else:
                ok = cls.from_json(payload).canon_hash() == event["payload_hash"]
            if not ok:
                raise ReceiptChainBroken(f"the payload of event {seq} does not match its hash", seq=seq)
        except ReceiptChainBroken:
            raise
        except Exception as exc:
            raise ReceiptChainBroken(f"event {seq} is malformed or altered ({type(exc).__name__})", seq=seq) from exc
        prev = event["event_hash"]
    return prev


__all__ = ["EventChain", "verify_events", "STATE_FORM", "state_hash", "unknown_state_hash", "value_digest",
           "json_value_digest", "action_value_digest", "bounded_id", "build_receipt", "receipt_ref", "ChainLink",
           "verify_chain", "ReceiptLedger", "verify_manifest_receipts"]

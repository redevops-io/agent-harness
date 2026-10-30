# SPDX-License-Identifier: AGPL-3.0-or-later
"""Schema identity strings shared with qa/v1, and the supported-schema table.

The neutral module mirrors the contracts ``qa/v1`` identifiers (they are strings, so nothing is imported). Contracts is
canonical; on any conflict the contracts value wins and this file is corrected, never the other way round.
"""
from __future__ import annotations

from types import MappingProxyType
from typing import Mapping, Tuple

from .errors import SchemaVersionUnsupported

#: identifier of this neutral interface (records, ports, canon-v1)
NEUTRAL_API = "qa-neutral/1"
#: the contracts family this module mirrors
CONTRACT_ID = "qa/v1"
#: hash algorithm ids: contracts ``rcv1`` and the neutral ``canon-v1`` give the same 64 hex digits
HASH_ALGORITHM = "canon-v1"
CONTRACT_HASH_ALGORITHM = "rcv1"
#: protocol api versions (the same strings as contracts)
ADAPTER_API_VERSION = "qa-adapter/1"
CHECK_ADAPTER_API_VERSION = "qa-check-adapter/1"
#: schema version carried by every qa/v1 schema
SCHEMA_VERSION = "0.1"

_P = "runtime-contracts/qa/"
SCHEMA_ID_TEST_SPEC = _P + "test-spec"
SCHEMA_ID_ASSERTION_SPEC = _P + "assertion-spec"
SCHEMA_ID_ASSERTION_SET = _P + "assertion-set"
SCHEMA_ID_OBSERVATION = _P + "observation"
SCHEMA_ID_OBSERVED_VALUE = _P + "observed-value"
SCHEMA_ID_ACTION_RECEIPT = _P + "action-receipt"
SCHEMA_ID_ASSERTION_RESULT = _P + "assertion-result"
SCHEMA_ID_FAULT_DETECTION = _P + "fault-detection"
SCHEMA_ID_FAULT_RECOVERY = _P + "fault-recovery"
SCHEMA_ID_FAULT_EVENT = _P + "fault-event"
SCHEMA_ID_CHECKPOINT = _P + "checkpoint"
SCHEMA_ID_REVIEW_ITEM = _P + "review-item"
SCHEMA_ID_REVIEW_RECORD = _P + "review-record"
SCHEMA_ID_RUN_MANIFEST = _P + "run-manifest"
SCHEMA_ID_POLICY_DECISION = _P + "policy-decision"
SCHEMA_ID_LOCATOR_CANDIDATE = _P + "locator-candidate"
SCHEMA_ID_CLEANUP_REPORT = _P + "cleanup-report"

#: schema id -> supported schema versions. Matching is exact: an unknown version (including a different major) is
#: refused, which is the fail-closed reading of "refuse an unknown major". ``qa-plan`` is deliberately absent: the
#: harness receives no QAPlan record (ADR A.2.4 item 7).
SUPPORTED_SCHEMAS: Mapping[str, Tuple[str, ...]] = MappingProxyType({
    sid: (SCHEMA_VERSION,)
    for sid in (
        SCHEMA_ID_TEST_SPEC, SCHEMA_ID_ASSERTION_SPEC, SCHEMA_ID_ASSERTION_SET, SCHEMA_ID_OBSERVATION,
        SCHEMA_ID_OBSERVED_VALUE, SCHEMA_ID_ACTION_RECEIPT, SCHEMA_ID_ASSERTION_RESULT, SCHEMA_ID_FAULT_DETECTION,
        SCHEMA_ID_FAULT_RECOVERY, SCHEMA_ID_FAULT_EVENT, SCHEMA_ID_CHECKPOINT, SCHEMA_ID_REVIEW_ITEM,
        SCHEMA_ID_REVIEW_RECORD, SCHEMA_ID_RUN_MANIFEST, SCHEMA_ID_POLICY_DECISION, SCHEMA_ID_LOCATOR_CANDIDATE,
        SCHEMA_ID_CLEANUP_REPORT,
    )
})


def require_supported(schema_id: str, schema_version: str) -> None:
    """Raise :class:`SchemaVersionUnsupported` unless (schema_id, schema_version) is in :data:`SUPPORTED_SCHEMAS`."""
    versions = SUPPORTED_SCHEMAS.get(schema_id) if isinstance(schema_id, str) else None
    if versions is None:
        raise SchemaVersionUnsupported(f"unsupported schema id {schema_id!r}", schema_id=schema_id)
    if not isinstance(schema_version, str) or schema_version not in versions:
        raise SchemaVersionUnsupported(
            f"unsupported schema version {schema_version!r} for {schema_id} (supported: {', '.join(versions)})",
            schema_id=schema_id, schema_version=schema_version,
        )

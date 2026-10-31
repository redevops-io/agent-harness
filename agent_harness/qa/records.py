# SPDX-License-Identifier: AGPL-3.0-or-later
"""Neutral plain-data records mirroring the contracts ``qa/v1`` schemas (non-normative mirror; contracts wins).

Every record and value is a frozen, keyword-only dataclass validated on construction (so constructing directly and
parsing JSON go through the same checks) with a strict ``from_json`` and a ``to_json``. Records carry ``schema_id`` and
``schema_version`` equal to the contracts strings and implement ``canonical_form()``: the identity fields only
(custody fields are excluded, ``None`` is absent, sets are sorted), with unknown additive fields kept in ``extras``
and included, so the hash of an object written by a newer same-version writer recomputes identically.
``LocatorCandidate`` is the exception: it is strict and has no ``extras`` (extra keys are rejected).

Data crossing the boundary is plain JSON only. Embedded hash references use the contracts spelling
(``rcv1:<64 hex>``, as in the golden patterns and vectors); hash *values produced by this module* are ``canon-v1:``
(see ``canon.py``). The harness receives no ``QAPlan`` record, so none is defined here.

Standard library only; imports nothing from any other package.
"""
from __future__ import annotations

import dataclasses
import enum
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from typing import (Annotated, Any, ClassVar, Dict, Iterable, Iterator, List, Optional, Tuple, Type, Union,
                    get_args, get_origin, get_type_hints)

from . import canon
from .errors import (CanonicalizationRefused, CheckProgramRefRequired, CoverageViolation, FaultBindingMismatch,
                     HashMismatch, QAError, RecoveryPayloadRejected, SpecInvalid, UnknownEnumValue, UnknownField)
from .schemas import (SCHEMA_ID_ACTION_RECEIPT, SCHEMA_ID_ASSERTION_RESULT, SCHEMA_ID_ASSERTION_SET,
                      SCHEMA_ID_ASSERTION_SPEC, SCHEMA_ID_CHECKPOINT, SCHEMA_ID_CLEANUP_REPORT,
                      SCHEMA_ID_FAULT_DETECTION, SCHEMA_ID_FAULT_EVENT, SCHEMA_ID_FAULT_RECOVERY,
                      SCHEMA_ID_LOCATOR_CANDIDATE, SCHEMA_ID_OBSERVATION, SCHEMA_ID_OBSERVED_VALUE,
                      SCHEMA_ID_POLICY_DECISION, SCHEMA_ID_REVIEW_ITEM, SCHEMA_ID_REVIEW_RECORD,
                      SCHEMA_ID_RUN_MANIFEST, SCHEMA_ID_TEST_SPEC, SCHEMA_VERSION, require_supported)

MAX_JSON_DEPTH = 32


# ---------------------------------------------------------------------------------------------------------------
# closed vocabularies (members and values equal the qa/v1 enums)
# ---------------------------------------------------------------------------------------------------------------

def _senum(name: str, *values: str, lower: bool = False) -> Type[enum.StrEnum]:
    cls = enum.StrEnum(name, {v: (v.lower() if lower else v) for v in values})
    cls.__module__ = __name__
    return cls


ActionKind = _senum("ActionKind", "CLICK", "FILL", "PRESS", "SELECT", "CHECK", "UNCHECK", "HOVER", "NAVIGATE",
                    "RESOLVE", lower=True)
CaptureKind = _senum("CaptureKind", "SCREENSHOT", "A11Y_SNAPSHOT", "DOM_META", "NETWORK_EXCERPT", "CONSOLE_EXCERPT",
                     "TRACE")
CheckTier = _senum("CheckTier", "OFFLINE", "K")
ClaimedClass = _senum("ClaimedClass", "BENIGN_DRIFT", "PRODUCT_DEFECT_SUSPECTED", "ENVIRONMENT", "UNKNOWN")
Determinism = _senum("Determinism", "DETERMINISTIC", "NON_DETERMINISTIC")
Executor = _senum("Executor", "PLAYWRIGHT", "SCRIPT")
FaultSignal = _senum("FaultSignal", "NONE", "LOCATOR_NOT_FOUND", "LOCATOR_AMBIGUOUS", "ACTION_TIMEOUT",
                     "TRANSITION_NOT_OBSERVED", "ASSERTION_FAILED", "ASSERTION_UNRESOLVABLE", "ENV_UNHEALTHY",
                     "POLICY_DENIED", "NETWORK_BLOCKED", "UNEXPECTED_OVERLAY", "SESSION_LOST")
Idempotency = _senum("Idempotency", "IDEMPOTENT", "AT_MOST_ONCE", "UNKNOWN")
LocateMode = _senum("LocateMode", "EXACT", "SEMANTIC")
LocatorStrategy = _senum("LocatorStrategy", "AUTHORED", "SEMANTIC_ALTERNATE", "MODEL_CANDIDATE")
Outcome = _senum("Outcome", "PASS", "PRODUCT_FAIL", "TEST_DEFECT", "ENVIRONMENT_FAIL", "INCONCLUSIVE")
PermittedRecovery = _senum("PermittedRecovery", "RETRY_SAME", "SEMANTIC_LOCATOR", "MODEL_CANDIDATE", "RESET_FIXTURE",
                           "CHECKPOINT_RESUME")
Phase = _senum("Phase", "BEFORE", "AFTER")
PolicyDecisionKind = _senum("PolicyDecisionKind", "ALLOW", "DENY", "REQUIRE_APPROVAL")
PostconditionResult = _senum("PostconditionResult", "SATISFIED", "NOT_SATISFIED", "NOT_EVALUATED")
PredicateOp = _senum("PredicateOp", "EQUALS", "NOT_EQUALS", "CONTAINS", "MATCHES", "EXISTS", "ABSENT", "LESS_THAN",
                     "GREATER_THAN")
RecoveryResult = _senum("RecoveryResult", "NOT_ATTEMPTED", "RECOVERED", "NOT_RECOVERED", "ABORTED")
RecoveryStrategy = _senum("RecoveryStrategy", "NONE", "RETRY_SAME", "SEMANTIC_LOCATOR", "MODEL_CANDIDATE",
                          "RESET_FIXTURE", "CHECKPOINT_RESUME", "ESCALATE_REVIEW")
ResetScope = _senum("ResetScope", "CONTEXT", "FIXTURE", "ENVIRONMENT")
Resolution = _senum("Resolution", "RESOLVED", "UNRESOLVED", "AMBIGUOUS")
ResourceKind = _senum("ResourceKind", "MANIFEST_PATH", "IMAGE_DIGEST", "NAMESPACE_OBJECT")
ResultCode = _senum("ResultCode", "OK", "LOCATOR_NOT_FOUND", "LOCATOR_AMBIGUOUS", "TARGET_NOT_ACTIONABLE",
                    "ACTION_TIMEOUT", "UNEXPECTED_DIALOG", "SESSION_LOST", "NAVIGATION_DENIED",
                    "ACTION_BLOCKED_BY_POLICY", "ENVIRONMENT_UNHEALTHY", "ADAPTER_CRASH", "BROWSER_CRASH",
                    "RESET_FAILED", "RESOURCE_LIMIT_EXCEEDED", "REDACTION_FAILED", "UNSUPPORTED_OPERATION")
RetrySafety = _senum("RetrySafety", "SAFE", "UNSAFE", "SAFE_WITH_IDEMPOTENCY_KEY")
ReviewDecision = _senum("ReviewDecision", "ACCEPT", "REJECT", "REQUEST_CHANGE")
SideEffect = _senum("SideEffect", "NONE", "READS_EXTERNAL", "WRITES_INTERNAL", "WRITES_EXTERNAL", "NOTIFIES_HUMAN",
                    "SPENDS_MONEY")
SkipReason = _senum("SkipReason", "NOT_REACHED", "PRECONDITION_UNMET", "POLICY_DENIED", "BUDGET_EXHAUSTED",
                    "SUBJECT_UNRESOLVED", "ENVIRONMENT", "HALTED_INTEGRITY")
SubjectKind = _senum("SubjectKind", "UI_ELEMENT", "URL", "RESPONSE", "RESOURCE_INPUTS")
SuppressedReason = _senum("SuppressedReason", "NONE", "NOT_PERMITTED", "NON_IDEMPOTENT", "POLICY_DENIED",
                          "BUDGET_EXHAUSTED", "KILL_SWITCH")
TargetKind = _senum("TargetKind", "UI", "RESOURCE", lower=True)
Verdict = _senum("Verdict", "PASS", "FAIL", "INDETERMINATE", "NOT_APPLICABLE")

ENUMS: Dict[str, Type[enum.StrEnum]] = {
    e.__name__: e for e in (
        ActionKind, CaptureKind, CheckTier, ClaimedClass, Determinism, Executor, FaultSignal, Idempotency,
        LocateMode, LocatorStrategy, Outcome, PermittedRecovery, Phase, PolicyDecisionKind, PostconditionResult,
        PredicateOp, RecoveryResult, RecoveryStrategy, ResetScope, Resolution, ResourceKind, ResultCode,
        RetrySafety, ReviewDecision, SideEffect, SkipReason, SubjectKind, SuppressedReason, TargetKind, Verdict,
    )
}


# ---------------------------------------------------------------------------------------------------------------
# immutable JSON
# ---------------------------------------------------------------------------------------------------------------

class FrozenMap(Mapping):
    """A hashable, read-only mapping (the frozen form of a JSON object)."""

    __slots__ = ("_d",)

    def __init__(self, data: Union[Mapping, Iterable] = ()) -> None:
        object.__setattr__(self, "_d", dict(data))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("FrozenMap is read-only")

    def __getitem__(self, key: Any) -> Any:
        return self._d[key]

    def __iter__(self) -> Iterator[Any]:
        return iter(self._d)

    def __len__(self) -> int:
        return len(self._d)

    def __hash__(self) -> int:
        return hash(frozenset(self._d.items()))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Mapping):
            return self._d == dict(other.items())
        return NotImplemented

    def __repr__(self) -> str:
        return f"FrozenMap({self._d!r})"


class Json:
    """Annotation marker: an arbitrary JSON value (no floats), deep-frozen on construction."""


def _freeze_json(value: Any, path: str, depth: int = 0) -> Any:
    if depth > MAX_JSON_DEPTH:
        raise SpecInvalid(f"{path}: JSON nested deeper than {MAX_JSON_DEPTH} levels")
    t = type(value)
    if value is None or t is bool or t is int:
        return value
    if t is str:
        _check_text(value, path)
        return value
    if isinstance(value, float):
        raise SpecInvalid(f"{path}: floats are refused; carry decimals as strings")
    if isinstance(value, Mapping):
        out = {}
        for k, v in value.items():
            if type(k) is not str:
                raise SpecInvalid(f"{path}: object keys must be strings")
            _check_text(k, path)
            out[k] = _freeze_json(v, f"{path}.{k}", depth + 1)
        return FrozenMap(out)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(v, f"{path}[{i}]", depth + 1) for i, v in enumerate(value))
    raise SpecInvalid(f"{path}: {t.__name__} is not a JSON value")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(v) for v in value]
    return value


def _check_text(text: str, path: str) -> None:
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        raise SpecInvalid(f"{path}: string is not valid Unicode text (lone surrogate)") from None


# ---------------------------------------------------------------------------------------------------------------
# field markers and type specs
# ---------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Pat:
    regex: str


@dataclass(frozen=True)
class Len:
    min: int = 0
    max: Optional[int] = None


@dataclass(frozen=True)
class Range:
    ge: Optional[int] = None
    le: Optional[int] = None


@dataclass(frozen=True)
class JsonMax:
    bytes: int


class Custody:
    """Marker: a custody field (who/when/how long); excluded from the canonical form."""


class SetOf:
    """Marker: an order-independent set; duplicates are refused and elements are sorted at construction."""


_ID = r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}"
Id = Annotated[str, Pat(_ID)]
Version = Annotated[str, Pat(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,63}")]
Rcv1Ref = Annotated[str, Pat(r"rcv1:[0-9a-f]{64}")]
Sha256Ref = Annotated[str, Pat(r"sha256:[0-9a-f]{64}")]
RcvOrSha = Annotated[str, Pat(r"(rcv1|sha256):[0-9a-f]{64}")]
SpecRefStr = Annotated[str, Pat(_ID + r"@rcv1:[0-9a-f]{64}")]
Reason = Annotated[str, Pat(r"[A-Z][A-Z0-9_]{0,63}")]
Hex7to64 = Annotated[str, Pat(r"[0-9a-f]{7,64}")]
Str512 = Annotated[str, Len(max=512)]
Str4096 = Annotated[str, Len(max=4096)]
Text1024 = Annotated[str, Len(min=1, max=1024)]
DecimalStr = Annotated[str, Pat(r"-?(0|[1-9][0-9]*)(\.[0-9]*[1-9])?")]
NonNegInt = Annotated[int, Range(ge=0)]


@dataclass
class TypeSpec:
    """Normalized description of one field type (also the introspection surface used by the parity tests)."""

    kind: str                                  # str | int | bool | enum | record | json | tuple
    optional: bool = False
    cls: Any = None                            # enum or record class
    pattern: Optional[str] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    minimum: Optional[int] = None
    maximum: Optional[int] = None
    max_json_bytes: Optional[int] = None
    elem: Optional["TypeSpec"] = None
    is_set: bool = False
    custody: bool = False
    compiled: Any = None


def _parse_type(tp: Any) -> TypeSpec:
    optional = False
    markers: List[Any] = []
    while True:
        origin = get_origin(tp)
        if origin is Annotated:
            args = get_args(tp)
            tp = args[0]
            markers.extend(args[1:])
            continue
        if origin is Union:
            rest = [a for a in get_args(tp) if a is not type(None)]
            if len(rest) != 1:
                raise TypeError(f"unsupported union {tp!r}")
            optional = True
            tp = rest[0]
            continue
        break
    origin = get_origin(tp)
    if tp is str:
        spec = TypeSpec("str")
    elif tp is int:
        spec = TypeSpec("int")
    elif tp is bool:
        spec = TypeSpec("bool")
    elif tp is Json:
        spec = TypeSpec("json")
    elif origin is tuple:
        args = get_args(tp)
        if len(args) != 2 or args[1] is not Ellipsis:
            raise TypeError(f"unsupported tuple type {tp!r}")
        spec = TypeSpec("tuple", elem=_parse_type(args[0]))
    elif isinstance(tp, type) and issubclass(tp, enum.Enum):
        spec = TypeSpec("enum", cls=tp)
    elif isinstance(tp, type) and issubclass(tp, _Value):
        spec = TypeSpec("record", cls=tp)
    else:
        raise TypeError(f"unsupported field type {tp!r}")
    spec.optional = optional
    for m in markers:
        if isinstance(m, Pat):
            spec.pattern = m.regex
            spec.compiled = re.compile(m.regex)
        elif isinstance(m, Len):
            spec.min_length = m.min if m.min else None
            spec.max_length = m.max
        elif isinstance(m, Range):
            spec.minimum, spec.maximum = m.ge, m.le
        elif isinstance(m, JsonMax):
            spec.max_json_bytes = m.bytes
        elif m is Custody:
            spec.custody = True
        elif m is SetOf:
            spec.is_set = True
    return spec


@dataclass
class FieldInfo:
    name: str
    required: bool
    custody: bool
    spec: TypeSpec


_FIELDS: Dict[type, List[FieldInfo]] = {}


def field_infos(cls: type) -> List[FieldInfo]:
    """Field descriptions of a record or value class (``extras`` excluded), in declaration order."""
    got = _FIELDS.get(cls)
    if got is None:
        hints = get_type_hints(cls, include_extras=True)
        got = []
        for f in dataclasses.fields(cls):
            if f.name == "extras":
                continue
            spec = _parse_type(hints[f.name])
            required = f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
            got.append(FieldInfo(f.name, required, spec.custody, spec))
        _FIELDS[cls] = got
    return got


# ---------------------------------------------------------------------------------------------------------------
# validation, decoding and export
# ---------------------------------------------------------------------------------------------------------------

def _set_key(elem: Any) -> str:
    if isinstance(elem, enum.Enum):
        return str(elem.value)
    if isinstance(elem, str):
        return unicodedata.normalize("NFC", elem)
    return canon.canonical_text(elem.canonical_form())


def _check(spec: TypeSpec, value: Any, path: str) -> Any:
    if value is None:
        if spec.optional:
            return None
        raise SpecInvalid(f"{path}: a value is required")
    kind = spec.kind
    if kind == "str":
        if type(value) is not str:
            raise SpecInvalid(f"{path}: expected a string, got {type(value).__name__}")
        _check_text(value, path)
        if spec.compiled is not None and spec.compiled.fullmatch(value) is None:
            raise SpecInvalid(f"{path}: does not match the required pattern {spec.pattern}")
        if spec.max_length is not None and len(value) > spec.max_length:
            raise SpecInvalid(f"{path}: longer than {spec.max_length} characters")
        if spec.min_length is not None and len(value) < spec.min_length:
            raise SpecInvalid(f"{path}: shorter than {spec.min_length} characters")
        return value
    if kind == "int":
        if type(value) is not int:
            raise SpecInvalid(f"{path}: expected an integer, got {type(value).__name__}")
        if spec.minimum is not None and value < spec.minimum:
            raise SpecInvalid(f"{path}: below the minimum {spec.minimum}")
        if spec.maximum is not None and value > spec.maximum:
            raise SpecInvalid(f"{path}: above the maximum {spec.maximum}")
        return value
    if kind == "bool":
        if type(value) is not bool:
            raise SpecInvalid(f"{path}: expected a boolean, got {type(value).__name__}")
        return value
    if kind == "enum":
        if isinstance(value, spec.cls):
            return value
        if type(value) is not str:
            raise SpecInvalid(f"{path}: expected one of {spec.cls.__name__}, got {type(value).__name__}")
        try:
            return spec.cls(value)
        except ValueError:
            raise UnknownEnumValue(f"{path}: {value!r} is not a {spec.cls.__name__} (unknown values fail closed)") from None
    if kind == "record":
        if not isinstance(value, spec.cls):
            raise SpecInvalid(f"{path}: expected {spec.cls.__name__}, got {type(value).__name__}")
        return value
    if kind == "json":
        frozen = _freeze_json(value, path)
        if spec.max_json_bytes is not None:
            try:
                size = len(canon.canonical_bytes(_thaw(frozen)))
            except CanonicalizationRefused as exc:
                raise SpecInvalid(f"{path}: {exc.message}") from None
            if size > spec.max_json_bytes:
                raise SpecInvalid(f"{path}: canonical JSON is {size} bytes, more than {spec.max_json_bytes}")
        return frozen
    if kind == "tuple":
        if not isinstance(value, (list, tuple)):
            raise SpecInvalid(f"{path}: expected a list, got {type(value).__name__}")
        items = [_check(spec.elem, v, f"{path}[{i}]") for i, v in enumerate(value)]
        if spec.is_set:
            keyed = sorted(((_set_key(e), e) for e in items), key=lambda p: p[0])
            keys = [k for k, _ in keyed]
            if len(set(keys)) != len(keys):
                raise SpecInvalid(f"{path}: duplicate elements in a set")
            items = [e for _, e in keyed]
        return tuple(items)
    raise AssertionError(kind)


def _decode(spec: TypeSpec, raw: Any, path: str) -> Any:
    """JSON -> value: objects become records, arrays become tuples, enum strings become members."""
    if raw is None:
        return None
    if spec.kind == "tuple":
        if not isinstance(raw, (list, tuple)):
            raise SpecInvalid(f"{path}: expected a JSON array")
        return tuple(_decode(spec.elem, v, f"{path}[{i}]") for i, v in enumerate(raw))
    if spec.kind == "record":
        if isinstance(raw, spec.cls):
            return raw
        if not isinstance(raw, Mapping):
            raise SpecInvalid(f"{path}: expected a JSON object")
        return spec.cls.from_json(raw, _path=path)
    return raw           # scalars, enums and Json are checked (and converted) by _check


def _plain(value: Any, identity: bool, with_hash: bool) -> Any:
    if isinstance(value, _Value):
        return value._export(identity, with_hash)
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, (tuple, list)):
        return [_plain(v, identity, with_hash) for v in value]
    if isinstance(value, Mapping):
        return {k: _plain(v, identity, with_hash) for k, v in value.items()}
    return value


@dataclass(frozen=True, kw_only=True)
class _Value:
    """Base of every neutral record and value."""

    extras: Mapping = FrozenMap()

    IS_RECORD: ClassVar[bool] = False
    SCHEMA_ID: ClassVar[Optional[str]] = None
    ALLOW_EXTRAS: ClassVar[bool] = True

    def __post_init__(self) -> None:
        cls = type(self)
        for info in field_infos(cls):
            value = _check(info.spec, getattr(self, info.name), f"{cls.__name__}.{info.name}")
            object.__setattr__(self, info.name, value)
        object.__setattr__(self, "extras", self._check_extras(self.extras))
        self._check_invariants()

    # -- validation hooks ----------------------------------------------------------------------------------
    def _check_extras(self, extras: Any) -> FrozenMap:
        cls = type(self)
        if extras is None:
            extras = {}
        if not isinstance(extras, Mapping):
            raise SpecInvalid(f"{cls.__name__}.extras: expected a mapping")
        if not extras:
            return FrozenMap()
        if not cls.ALLOW_EXTRAS:
            raise RecoveryPayloadRejected(f"{cls.__name__} is strict: unknown fields {sorted(map(str, extras))} rejected")
        known = {i.name for i in field_infos(cls)}
        if cls.IS_RECORD:
            known |= {"schema_id", "schema_version", "content_hash", "custody"}
        out = {}
        for k, v in extras.items():
            if type(k) is not str:
                raise SpecInvalid(f"{cls.__name__}.extras: keys must be strings")
            if k in known:
                raise SpecInvalid(f"{cls.__name__}.extras: {k!r} is a known field")
            out[k] = _freeze_json(v, f"{cls.__name__}.extras.{k}")
        return FrozenMap(out)

    def _check_invariants(self) -> None:
        """Cross-field rules of the semantics; overridden per class."""

    # -- export ----------------------------------------------------------------------------------------------
    def _export(self, identity: bool, with_hash: bool) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        custody: Dict[str, Any] = {}
        if self.IS_RECORD:
            out["schema_id"] = self.SCHEMA_ID
            out["schema_version"] = SCHEMA_VERSION
        for info in field_infos(type(self)):
            value = getattr(self, info.name)
            if value is None:
                continue
            if info.custody:
                if not identity:
                    custody[info.name] = _plain(value, identity, with_hash)
                continue
            out[info.name] = _plain(value, identity, with_hash)
        for k, v in self.extras.items():
            out[k] = _plain(v, identity, with_hash)
        if not identity:
            if custody:
                out["custody"] = custody
            if with_hash and self.IS_RECORD:
                out["content_hash"] = self.canon_hash()
        return out

    def canonical_form(self) -> Dict[str, Any]:
        """The identity payload that is hashed: plain JSON, custody excluded, ``None`` absent, extras included."""
        return self._export(True, False)

    def canon_hash(self) -> str:
        """``canon-v1:<64 hex>`` of :meth:`canonical_form` (the contracts spelling is ``rcv1:`` + the same digits)."""
        return canon.canon_hash(self.canonical_form())

    def to_json(self, *, with_hash: bool = False) -> Dict[str, Any]:
        """Plain JSON including custody (under ``custody``); ``with_hash`` adds ``content_hash`` to each record."""
        return self._export(False, with_hash)

    @classmethod
    def from_json(cls, data: Any, *, _path: Optional[str] = None):
        path = _path or cls.__name__
        if not isinstance(data, Mapping):
            raise SpecInvalid(f"{path}: expected a JSON object, got {type(data).__name__}")
        infos = {i.name: i for i in field_infos(cls)}
        kwargs: Dict[str, Any] = {}
        extras: Dict[str, Any] = {}
        stored_hash = None
        custody_in = None
        for key, raw in data.items():
            if type(key) is not str:
                raise SpecInvalid(f"{path}: object keys must be strings")
            info = infos.get(key)
            if info is not None:
                if info.custody:
                    raise SpecInvalid(f"{path}.{key}: custody fields belong under 'custody', not at the top level")
                kwargs[key] = _decode(info.spec, raw, f"{path}.{key}")
            elif cls.IS_RECORD and key == "schema_id":
                if raw != cls.SCHEMA_ID:
                    raise SpecInvalid(f"{path}: schema_id {raw!r} is not {cls.SCHEMA_ID}")
            elif cls.IS_RECORD and key == "schema_version":
                require_supported(cls.SCHEMA_ID, raw)
            elif cls.IS_RECORD and key == "content_hash":
                stored_hash = raw
            elif cls.IS_RECORD and key == "custody":
                custody_in = raw
            elif cls.ALLOW_EXTRAS:
                extras[key] = raw
            else:
                raise UnknownField(f"{path}: unknown field {key!r}")
        if custody_in is not None:
            if not isinstance(custody_in, Mapping):
                raise SpecInvalid(f"{path}.custody: expected a JSON object")
            for key, raw in custody_in.items():
                info = infos.get(key) if type(key) is str else None
                if info is None:
                    raise UnknownField(f"{path}.custody: unknown custody field {key!r}")
                if not info.custody:
                    raise SpecInvalid(f"{path}.custody.{key}: {key!r} is an identity field, not a custody field")
                kwargs[key] = _decode(info.spec, raw, f"{path}.custody.{key}")
        for name, info in infos.items():
            if info.required and name not in kwargs:
                raise SpecInvalid(f"{path}.{name}: required field is missing")
        obj = cls(**kwargs, extras=extras)
        if stored_hash is not None:
            try:
                same = canon.same_digest(stored_hash, obj.canon_hash())
            except CanonicalizationRefused:
                raise SpecInvalid(f"{path}.content_hash: not a canon-v1 or rcv1 hash") from None
            if not same:
                raise HashMismatch(f"{path}: stored content_hash does not match the recomputed hash")
        return obj


def _value(cls):
    return dataclass(frozen=True, kw_only=True)(cls)


def _record(schema_id: str):
    def deco(cls):
        cls.IS_RECORD = True
        cls.SCHEMA_ID = schema_id
        return dataclass(frozen=True, kw_only=True)(cls)
    return deco


# ---------------------------------------------------------------------------------------------------------------
# reused contract types (Check, EvidenceRef, SecretRef) and small values
# ---------------------------------------------------------------------------------------------------------------

# The inventory types the optional strings of these three reused contract types as ``str`` (not ``Optional[str]``) and
# the golden vectors omit them when unset. The neutral reading: absent is ``None`` and omitted from the canonical
# form, and an explicit empty string is refused, so "" can never mean two things.
OptText = Optional[Annotated[str, Len(min=1)]]


@_value
class Check(_Value):
    check_id: str
    verdict: Verdict
    determinism: Determinism = Determinism.NON_DETERMINISTIC   # fail closed when absent
    detail: OptText = None
    evidence_refs: Tuple[str, ...] = ()


@_value
class EvidenceRef(_Value):
    ref: str
    content_hash: OptText = None
    media_type: OptText = None
    modality: OptText = None
    ref_type: OptText = None
    source: OptText = None
    version: OptText = None


@_value
class SecretRef(_Value):
    provider: str
    key: Optional[str] = None
    namespace: OptText = None
    path: OptText = None
    version: Optional[str] = None


@_value
class ComponentRef(_Value):
    id: Id
    version: Version


@_value
class ToolRef(_Value):
    name: Id
    version: Version


@_value
class ModelRef(_Value):
    model_id: Str512
    provider: Id
    parameters: Optional[Annotated[Json, JsonMax(4096)]] = None


@_value
class ObservationSource(_Value):
    adapter: Id
    adapter_version: Version
    browser: Optional[Str512] = None
    browser_version: Optional[Str512] = None


@_value
class VersionPin(_Value):
    component: Id
    version: Version
    revision: Optional[Hex7to64] = None


@_value
class Counters(_Value):
    cost: Optional[DecimalStr] = None
    infrastructure_ms: NonNegInt = 0
    model_calls: NonNegInt = 0
    tokens: NonNegInt = 0


@_value
class RecoveryCost(_Value):
    model_calls: NonNegInt = 0
    ms: NonNegInt = 0
    tokens: NonNegInt = 0


@_value
class LocatorDescriptor(_Value):
    frame_path: Tuple[Str512, ...] = ()
    label: Optional[Str512] = None
    name: Optional[Str512] = None
    role: Optional[Str512] = None
    test_id: Optional[Str512] = None
    text_hint: Optional[Str512] = None


@_value
class ResourceRef(_Value):
    kind: ResourceKind
    api_version: Optional[Str512] = None
    digest: Optional[Sha256Ref] = None
    name: Optional[Str512] = None
    namespace: Optional[Str512] = None
    object_kind: Optional[Str512] = None
    path: Optional[Annotated[str, Len(min=1, max=512)]] = None

    def _check_invariants(self) -> None:
        # each kind carries exactly its typed fields; anything else makes the identity ambiguous
        if self.kind is ResourceKind.MANIFEST_PATH:
            need, allowed = {"path"}, {"path"}
        elif self.kind is ResourceKind.IMAGE_DIGEST:
            need, allowed = {"digest"}, {"digest"}
        else:
            need = {"api_version", "object_kind", "name"}
            allowed = need | {"namespace"}
        present = {n for n in ("api_version", "digest", "name", "namespace", "object_kind", "path")
                   if getattr(self, n) is not None}
        if need - present:
            raise SpecInvalid(f"ResourceRef({self.kind}): missing {sorted(need - present)}")
        if present - allowed:
            raise SpecInvalid(f"ResourceRef({self.kind}): fields {sorted(present - allowed)} do not belong to this kind")


@_value
class CheckProgramRef(_Value):
    id: Id
    program_hash: Sha256Ref
    version: Version


@_value
class CheckProfile(_Value):
    case_root_digest: Sha256Ref
    tier: CheckTier
    disposable_proof_ref: Optional[RcvOrSha] = None

    def _check_invariants(self) -> None:
        if self.tier is CheckTier.K and self.disposable_proof_ref is None:
            raise SpecInvalid("CheckProfile: disposable_proof_ref is required when tier is K")


@_value
class CheckSubject(_Value):
    check_program_ref: CheckProgramRef
    inputs: Tuple[ResourceRef, ...]
    tier: CheckTier


@_value
class Action(_Value):
    kind: ActionKind
    secret_ref: Optional[SecretRef] = None
    value: Optional[Str512] = None

    def _check_invariants(self) -> None:
        if self.value is not None and self.secret_ref is not None:
            raise SpecInvalid("Action: give either a value or a secret_ref, not both")


@_value
class ActionOutcome(_Value):
    result: ResultCode
    detail: Optional[Str512] = None


@_value
class ActionRecord(_Value):
    kind: ActionKind
    value_digest: Optional[Rcv1Ref] = None


@_value
class ActionRequest(_Value):
    action: Action
    idempotency_key: Id
    policy_decision_ref: Rcv1Ref
    step_id: Id


@_value
class AssertionSubject(_Value):
    kind: SubjectKind
    inputs: Tuple[ResourceRef, ...] = ()
    locator: Optional[LocatorDescriptor] = None
    pattern: Optional[Str512] = None
    subject_alternates: Tuple[LocatorDescriptor, ...] = ()


@_value
class Budget(_Value):
    deadline_ms: Annotated[int, Range(ge=1, le=86400000)]
    max_model_calls: Annotated[int, Range(ge=0, le=100000)] = 0
    max_recovery_attempts: Annotated[int, Range(ge=0, le=1000)] = 0
    max_retries: Annotated[int, Range(ge=0, le=1000)] = 0
    max_tokens: Annotated[int, Range(ge=0, le=100000000)] = 0


@_value
class FixtureRef(_Value):
    content_hash: RcvOrSha
    id: Id


@_value
class LocateCandidate(_Value):
    candidate_ref: Id
    enabled: bool
    unique: bool
    visible: bool
    name: Optional[Str512] = None
    role: Optional[Str512] = None


@_value
class LocateResult(_Value):
    candidates: Tuple[LocateCandidate, ...] = ()


@_value
class LocatorUsed(_Value):
    descriptor: LocatorDescriptor
    strategy: LocatorStrategy


@_value
class ObserveRequest(_Value):
    phase: Phase
    step_id: Id
    capture: Annotated[Tuple[CaptureKind, ...], SetOf] = ()


@_value
class Postcondition(_Value):
    declared: Text1024
    result: PostconditionResult
    observed: Optional[Str512] = None


@_value
class Predicate(_Value):
    op: PredicateOp
    expected: Optional[Annotated[Json, JsonMax(4096)]] = None


@_value
class ResetResult(_Value):
    ok: bool
    scope: ResetScope


@_value
class ResolvedResource(_Value):
    content_digest: Sha256Ref
    ref: ResourceRef


@_value
class ResolvedTarget(_Value):
    candidate_ref: Id
    locator: LocatorDescriptor
    strategy: LocatorStrategy


@_value
class SessionHandle(_Value):
    session_id: Id


@_value
class SessionProfile(_Value):
    egress_profile: Id
    redaction_policy_id: RcvOrSha
    allow_downloads: bool = False
    allow_file_choosers: bool = False
    allow_popups: bool = False


@_value
class SkippedAssertion(_Value):
    assertion_id: Id
    reason_code: SkipReason


@_value
class StepTarget(_Value):
    kind: TargetKind
    locator: Optional[LocatorDescriptor] = None
    resource: Optional[ResourceRef] = None

    def _check_invariants(self) -> None:
        if self.kind is TargetKind.UI:
            if self.locator is None or self.resource is not None:
                raise SpecInvalid("StepTarget(ui) needs a locator and no resource")
        else:
            if self.resource is None or self.locator is not None:
                raise SpecInvalid("StepTarget(resource) needs a resource and no locator")


@_value
class Step(_Value):
    step_id: Id
    intent: Text1024
    target: StepTarget
    action: Action
    expected_transition: Text1024
    observation_deadline_ms: Annotated[int, Range(ge=1, le=3600000)]
    side_effect: SideEffect
    idempotency: Idempotency = Idempotency.UNKNOWN              # UNKNOWN means no blind retry
    retry_safety: RetrySafety = RetrySafety.UNSAFE              # fail closed when absent
    permitted_recovery: Annotated[Tuple[PermittedRecovery, ...], SetOf] = ()


@_value
class StepResult(_Value):
    step_id: Id
    result: ResultCode
    postcondition: PostconditionResult
    receipts: Tuple[Rcv1Ref, ...]


@_value
class TestTarget(_Value):
    __test__ = False        # not a pytest test class

    app_id: Id
    build_sha: Hex7to64
    environment_profile: Id


@_value
class WorkspaceHandle(_Value):
    workspace_id: Id


# ---------------------------------------------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------------------------------------------

@_record(SCHEMA_ID_ASSERTION_SPEC)
class AssertionSpec(_Value):
    assertion_id: Id
    executor: Executor
    subject: AssertionSubject
    predicate: Predicate
    verifier_id: Id
    verifier_version_range: Annotated[str, Pat(r"[A-Za-z0-9.<>=!~,*+ -]{1,64}")]
    check_program_ref: Optional[CheckProgramRef] = None
    determinism: Determinism = Determinism.DETERMINISTIC
    on_unresolvable: Verdict = Verdict.INDETERMINATE
    requirement_ref: Optional[Annotated[str, Pat(r"[A-Za-z0-9._:/-]{1,128}")]] = None

    def _check_invariants(self) -> None:
        if self.determinism is not Determinism.DETERMINISTIC:
            raise SpecInvalid("AssertionSpec: a gating assertion must be DETERMINISTIC")
        if self.on_unresolvable is not Verdict.INDETERMINATE:
            raise SpecInvalid("AssertionSpec: on_unresolvable is fixed to INDETERMINATE")
        if self.executor is Executor.SCRIPT:
            if self.check_program_ref is None:
                raise CheckProgramRefRequired("AssertionSpec: executor SCRIPT requires check_program_ref")
        elif self.check_program_ref is not None:
            raise SpecInvalid("AssertionSpec: check_program_ref is only for executor SCRIPT")


@_record(SCHEMA_ID_TEST_SPEC)
class TestSpec(_Value):
    __test__ = False        # not a pytest test class

    id: Id
    version: Annotated[int, Range(ge=1)]
    target: TestTarget
    steps: Tuple[Step, ...]
    assertions: Annotated[Tuple[AssertionSpec, ...], SetOf]
    budget: Budget
    egress_profile: Id
    allowed_side_effects: Annotated[Tuple[SideEffect, ...], SetOf] = ()
    cleanup: Tuple[Str512, ...] = ()
    depends_on: Annotated[Tuple[SpecRefStr, ...], SetOf] = ()
    fixtures: Annotated[Tuple[FixtureRef, ...], SetOf] = ()
    preconditions: Tuple[Str512, ...] = ()
    sensitivity_labels: Annotated[Tuple[Id, ...], SetOf] = ()
    author: Annotated[Optional[Str512], Custody] = None
    approved_at: Annotated[Optional[Str512], Custody] = None
    reviewer: Annotated[Optional[Str512], Custody] = None

    def _check_invariants(self) -> None:
        ids = [a.assertion_id for a in self.assertions]
        if len(set(ids)) != len(ids):
            raise SpecInvalid("TestSpec: assertion ids must be unique")
        step_ids = [s.step_id for s in self.steps]
        if len(set(step_ids)) != len(step_ids):
            raise SpecInvalid("TestSpec: step ids must be unique")

    def spec_hash(self) -> str:
        """``canon-v1:<hex>`` of the spec (custody excluded)."""
        return self.canon_hash()

    def spec_ref(self) -> str:
        """``<id>@rcv1:<hex>``, the spec reference used by ``depends_on`` and ``RunManifest.spec_ref``."""
        return f"{self.id}@{canon.to_contract_ref(self.spec_hash())}"

    def assertion_set_hash(self) -> str:
        return assertion_set_hash(self.assertions)


@_record(SCHEMA_ID_OBSERVATION)
class Observation(_Value):
    observation_id: Id
    step_id: Id
    phase: Phase
    source: ObservationSource
    redaction_applied: bool
    redaction_policy_id: RcvOrSha
    url: Optional[Annotated[str, Len(min=1, max=2048)]] = None
    frame: Optional[Str512] = None
    a11y_snapshot_ref: Optional[EvidenceRef] = None
    console_excerpt_ref: Optional[EvidenceRef] = None
    network_excerpt_ref: Optional[EvidenceRef] = None
    screenshot_ref: Optional[EvidenceRef] = None
    dom_meta: Optional[Annotated[Json, JsonMax(4096)]] = None
    duration_ms: Annotated[Optional[NonNegInt], Custody] = None
    observed_at: Annotated[Optional[Str512], Custody] = None


@_record(SCHEMA_ID_OBSERVED_VALUE)
class ObservedValue(_Value):
    resolution: Resolution
    match_count: NonNegInt
    observation_id: Optional[Id] = None
    value: Optional[Annotated[Json, JsonMax(16384)]] = None
    value_digest: Optional[Rcv1Ref] = None


@_record(SCHEMA_ID_ACTION_RECEIPT)
class ActionReceipt(_Value):
    receipt_id: Id
    run_id: Id
    step_id: Id
    attempt: Annotated[int, Range(ge=1)]
    tool: ToolRef
    action: ActionRecord
    idempotency_key: Id
    policy_decision_ref: Rcv1Ref
    postcondition: Postcondition
    result: ResultCode
    state_before_hash: Rcv1Ref
    state_after_hash: Rcv1Ref
    execution_binding: Optional[Rcv1Ref] = None
    locator_used: Optional[LocatorUsed] = None
    resource_used: Optional[ResourceRef] = None
    duration_ms: Annotated[Optional[NonNegInt], Custody] = None
    recovery_rationale: Annotated[Optional[Str4096], Custody] = None


@_record(SCHEMA_ID_ASSERTION_RESULT)
class AssertionResult(_Value):
    assertion_id: Id
    assertion_set_hash: Rcv1Ref
    expected_hash: Rcv1Ref
    executor: Executor
    check: Check
    verifier: ComponentRef
    actual: Optional[Annotated[Json, JsonMax(16384)]] = None
    actual_digest: Optional[Rcv1Ref] = None
    observation_id: Optional[Id] = None
    reason_code: Optional[Reason] = None
    evaluated_at: Annotated[Optional[Str512], Custody] = None


@_record(SCHEMA_ID_FAULT_DETECTION)
class FaultDetection(_Value):
    detected: bool
    signal: FaultSignal
    claimed_class: ClaimedClass
    detector: ComponentRef
    evidence_refs: Annotated[Tuple[EvidenceRef, ...], SetOf] = ()
    detected_at_ms_into_step: Annotated[Optional[NonNegInt], Custody] = None


@_record(SCHEMA_ID_FAULT_RECOVERY)
class FaultRecovery(_Value):
    detection_ref: Rcv1Ref
    attempted: bool
    strategy: RecoveryStrategy
    result: RecoveryResult
    postcondition_verified: bool
    attempts: NonNegInt = 0
    candidates_considered: NonNegInt = 0
    candidates_rejected_reasons: Tuple[Reason, ...] = ()
    suppressed_reason: SuppressedReason = SuppressedReason.NONE
    cost: Annotated[Optional[RecoveryCost], Custody] = None

    def _check_invariants(self) -> None:
        # the result comes from the step's own postcondition check, never from the run outcome
        if self.result is RecoveryResult.RECOVERED and not self.postcondition_verified:
            raise SpecInvalid("FaultRecovery: RECOVERED requires postcondition_verified")


@_record(SCHEMA_ID_FAULT_EVENT)
class FaultEvent(_Value):
    fault_event_id: Id
    step_id: Id
    sequence: NonNegInt
    detection: FaultDetection
    recovery: FaultRecovery

    def _check_invariants(self) -> None:
        expected = canon.to_contract_ref(self.detection.canon_hash())
        if self.recovery.detection_ref != expected:
            raise FaultBindingMismatch("FaultEvent: recovery.detection_ref is not the hash of the detection it responds to")


@_record(SCHEMA_ID_CHECKPOINT)
class Checkpoint(_Value):
    checkpoint_id: Id
    run_id: Id
    step_index: NonNegInt
    state_hash: Rcv1Ref
    fixture_snapshot_id: Optional[Id] = None
    taken_at: Annotated[Optional[Str512], Custody] = None


@_record(SCHEMA_ID_REVIEW_ITEM)
class ReviewItem(_Value):
    review_item_id: Id
    run_id: Id
    reason: Reason
    evidence_refs: Annotated[Tuple[EvidenceRef, ...], SetOf] = ()
    suggested_change_ref: Optional[Rcv1Ref] = None
    created_at: Annotated[Optional[Str512], Custody] = None


@_record(SCHEMA_ID_REVIEW_RECORD)
class ReviewRecord(_Value):
    review_item_ref: Rcv1Ref
    decision: ReviewDecision
    decided_at: Annotated[Optional[Str512], Custody] = None
    review_minutes: Annotated[Optional[NonNegInt], Custody] = None
    reviewer_id: Annotated[Optional[Str512], Custody] = None


@_record(SCHEMA_ID_CLEANUP_REPORT)
class CleanupReport(_Value):
    handle_id: Id
    verified: bool
    leaked: Annotated[Tuple[Id, ...], SetOf] = ()
    released: Annotated[Tuple[Id, ...], SetOf] = ()
    closed_at: Annotated[Optional[Str512], Custody] = None


@_record(SCHEMA_ID_POLICY_DECISION)
class PolicyDecisionRecord(_Value):
    """Neutral mirror of the contracts ``QAPolicyDecision`` (same schema id)."""

    policy_decision_id: Id
    run_id: Id
    step_id: Id
    principal: Str512
    app: Id
    decision: PolicyDecisionKind
    provider: Id
    action_kind: ActionKind
    side_effect: SideEffect
    phase: Annotated[str, Len(max=32)] = "ui_action"
    rule_id: Optional[Id] = None
    scope: Annotated[Tuple[Str512, ...], SetOf] = ()
    decided_at: Annotated[Optional[Str512], Custody] = None
    reason: Annotated[Optional[Str4096], Custody] = None


@_record(SCHEMA_ID_LOCATOR_CANDIDATE)
class LocatorCandidate(_Value):
    """The only thing a recoverer may return. Strict: any extra key is rejected, there is no ``extras``."""

    strategy: LocatorStrategy
    label: Optional[Str512] = None
    name: Optional[Str512] = None
    role: Optional[Str512] = None
    test_id: Optional[Str512] = None

    ALLOW_EXTRAS: ClassVar[bool] = False

    @classmethod
    def from_json(cls, data: Any, *, _path: Optional[str] = None):
        try:
            return super().from_json(data, _path=_path)
        except RecoveryPayloadRejected:
            raise
        except QAError as exc:
            raise RecoveryPayloadRejected(f"recovery payload rejected: {exc.message}", cause_code=exc.code) from exc


@_record(SCHEMA_ID_RUN_MANIFEST)
class RunManifest(_Value):
    run_id: Id
    spec_ref: SpecRefStr
    adapter: ComponentRef
    assertion_set_hash: Rcv1Ref
    assertions_declared: Annotated[Tuple[Id, ...], SetOf]
    assertions_executed: Annotated[Tuple[Id, ...], SetOf]
    assertions_skipped: Annotated[Tuple[SkippedAssertion, ...], SetOf]
    egress_policy_id: RcvOrSha
    redaction_policy_id: RcvOrSha
    model_recovery_enabled: bool
    recovery_enabled: bool
    outcome: Outcome
    trace_id: Annotated[str, Pat(r"[0-9a-f]{32}")]
    assertion_results: Annotated[Tuple[AssertionResult, ...], SetOf] = ()
    browser: Optional[Str512] = None
    fault_events: Tuple[FaultEvent, ...] = ()
    model: Optional[ModelRef] = None
    outcome_reason: Optional[Reason] = None
    plan_hash: Optional[Rcv1Ref] = None
    planner: Optional[ComponentRef] = None
    resource_limits: Optional[Annotated[Json, JsonMax(4096)]] = None
    sandbox_image_digest: Optional[Sha256Ref] = None
    seeds: Tuple[int, ...] = ()
    step_results: Annotated[Tuple[StepResult, ...], SetOf] = ()
    versions: Annotated[Tuple[VersionPin, ...], SetOf] = ()
    counters: Annotated[Optional[Counters], Custody] = None
    finished_at: Annotated[Optional[Str512], Custody] = None
    started_at: Annotated[Optional[Str512], Custody] = None

    def accounting_ok(self) -> bool:
        """declared == executed + skipped, executed and skipped disjoint, executed == ids with a result."""
        declared = set(self.assertions_declared)
        executed = set(self.assertions_executed)
        skipped = {s.assertion_id for s in self.assertions_skipped}
        results = [r.assertion_id for r in self.assertion_results]
        return (executed.isdisjoint(skipped) and (executed | skipped) == declared
                and len(skipped) == len(self.assertions_skipped)
                and len(set(results)) == len(results) and set(results) == executed)

    def _check_invariants(self) -> None:
        if not self.accounting_ok():
            # a manifest whose accounting is broken can only be recorded as INCONCLUSIVE(ASSERTION_INTEGRITY)
            if not (self.outcome is Outcome.INCONCLUSIVE and self.outcome_reason == "ASSERTION_INTEGRITY"):
                raise CoverageViolation("RunManifest: declared != executed + skipped (disjoint); only "
                                        "INCONCLUSIVE(ASSERTION_INTEGRITY) may be recorded")
        if self.outcome is Outcome.PASS and self.assertions_skipped:
            raise CoverageViolation("RunManifest: a PASS requires no skipped assertion")


# ---------------------------------------------------------------------------------------------------------------
# assertion-set document (the frozen-assertion hash input)
# ---------------------------------------------------------------------------------------------------------------

def assertion_set_document(assertions: Iterable[AssertionSpec]) -> Dict[str, Any]:
    """The document hashed as ``assertion_set_hash``: sorted canonical assertions under the assertion-set schema."""
    items = list(assertions)
    for a in items:
        if not isinstance(a, AssertionSpec):
            raise SpecInvalid("assertion_set_document: expected AssertionSpec objects")
    ids = [a.assertion_id for a in items]
    if len(set(ids)) != len(ids):
        raise SpecInvalid("assertion_set_document: assertion ids must be unique")
    forms = sorted((a.canonical_form() for a in items), key=canon.canonical_text)
    return {"assertions": forms, "schema_id": SCHEMA_ID_ASSERTION_SET, "schema_version": SCHEMA_VERSION}


def assertion_set_hash(assertions: Iterable[AssertionSpec]) -> str:
    """``canon-v1:<hex>`` of the assertion-set document (order-independent)."""
    return canon.canon_hash(assertion_set_document(assertions))


# ---------------------------------------------------------------------------------------------------------------
# registries
# ---------------------------------------------------------------------------------------------------------------

RECORD_TYPES: Tuple[type, ...] = (
    ActionReceipt, AssertionResult, AssertionSpec, Checkpoint, CleanupReport, FaultDetection, FaultEvent,
    FaultRecovery, LocatorCandidate, Observation, ObservedValue, PolicyDecisionRecord, ReviewItem, ReviewRecord,
    RunManifest, TestSpec,
)
VALUE_TYPES: Tuple[type, ...] = (
    Action, ActionOutcome, ActionRecord, ActionRequest, AssertionSubject, Budget, Check, CheckProfile,
    CheckProgramRef, CheckSubject, ComponentRef, Counters, EvidenceRef, FixtureRef, LocateCandidate, LocateResult,
    LocatorDescriptor, LocatorUsed, ModelRef, ObservationSource, ObserveRequest, Postcondition, Predicate,
    RecoveryCost, ResetResult, ResolvedResource, ResolvedTarget, ResourceRef, SecretRef, SessionHandle,
    SessionProfile, SkippedAssertion, Step, StepResult, StepTarget, TestTarget, ToolRef, VersionPin,
    WorkspaceHandle,
)
RECORDS_BY_SCHEMA_ID: Dict[str, type] = {r.SCHEMA_ID: r for r in RECORD_TYPES}

__all__ = sorted(
    [e for e in ENUMS]
    + [c.__name__ for c in RECORD_TYPES + VALUE_TYPES]
    + ["FrozenMap", "Json", "ENUMS", "RECORD_TYPES", "VALUE_TYPES", "RECORDS_BY_SCHEMA_ID", "FieldInfo", "TypeSpec",
       "field_infos", "assertion_set_document", "assertion_set_hash"]
)

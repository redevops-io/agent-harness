# SPDX-License-Identifier: AGPL-3.0-or-later
"""Read-only ports the verifier reads frozen facts from: the spec store and the check-program registry.

The spec store is content-addressed by ``assertion_set_hash`` and has exactly one
operation, a read. There is no ``put``, no ``delete`` and no handle through which a stored assertion could be changed:
the writer (the bridge-side loader) is a different component and never reaches the verifier, the adapter or a
recoverer. The two implementations here are test and reference implementations; both are read-only by construction.

Standard library and neutral-module imports only.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable

from . import canon
from .errors import CanonicalizationRefused, SpecInvalid
from .schemas import SCHEMA_ID_ASSERTION_SET, require_supported

#: a stored assertion set larger than this is refused unread (a spec store holds small documents)
MAX_SET_BYTES = 4 * 1024 * 1024


@runtime_checkable
class SpecStore(Protocol):
    """Content-addressed, read-only store of frozen assertion sets.

    ``get_assertion_set(h)`` returns the JSON assertion documents stored under the hash ``h`` (``rcv1:<hex>`` or
    ``canon-v1:<hex>``; the 64 hex digits are the key) and raises ``KeyError`` if there is none. It returns what is
    stored, unchecked: the verifier, not the store, decides whether it matches the pin.
    """

    def get_assertion_set(self, assertion_set_hash: str) -> Sequence[Mapping[str, Any]]: ...


@runtime_checkable
class ProgramRegistry(Protocol):
    """Read-only registry of check programs (registry entries are individually hashed).

    ``registered_program_hash(id, version)`` returns the full ``sha256:<64 hex>`` of the program source the registry
    resolves for ``{id, version}``, or ``None`` if nothing is registered under that name.
    """

    def registered_program_hash(self, program_id: str, version: str) -> Optional[str]: ...


def _key(assertion_set_hash: Any) -> str:
    try:
        return canon.digest_of(assertion_set_hash)
    except CanonicalizationRefused:
        raise KeyError("not a canon-v1 or rcv1 hash") from None


class InMemorySpecStore:
    """Immutable in-memory store. Documents are kept as JSON text and a fresh copy is returned on every read, so a
    caller can neither change what is stored nor see another caller's changes."""

    __slots__ = ("_docs",)

    def __init__(self, sets: Optional[Mapping[str, Sequence[Mapping[str, Any]]]] = None) -> None:
        docs: Dict[str, str] = {}
        for h, assertions in (sets or {}).items():
            docs[_key(h)] = json.dumps(list(assertions), ensure_ascii=False, sort_keys=True)
        object.__setattr__(self, "_docs", docs)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("InMemorySpecStore is read-only")

    def get_assertion_set(self, assertion_set_hash: str) -> Sequence[Mapping[str, Any]]:
        return json.loads(self._docs[_key(assertion_set_hash)])


def _no_duplicate_keys(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise SpecInvalid(f"duplicate key {k!r} in a stored assertion document")
        out[k] = v
    return out


class DirectorySpecStore:
    """Reads ``<root>/<64 hex>.json``: either a JSON array of assertions or an ``assertion-set`` document. Opens files
    for binary reading only; the directory is meant to be mounted read-only in the worker."""

    __slots__ = ("_root",)

    def __init__(self, root: Any) -> None:
        object.__setattr__(self, "_root", Path(root))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("DirectorySpecStore is read-only")

    def get_assertion_set(self, assertion_set_hash: str) -> Sequence[Mapping[str, Any]]:
        path = self._root / (_key(assertion_set_hash) + ".json")     # the name is 64 validated hex digits: no traversal
        try:
            with open(path, "rb") as handle:
                raw = handle.read(MAX_SET_BYTES + 1)
        except FileNotFoundError:
            raise KeyError("no assertion set stored under this hash") from None
        if len(raw) > MAX_SET_BYTES:
            raise SpecInvalid("stored assertion set is larger than the allowed size")
        try:
            doc = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicate_keys)
        except (UnicodeDecodeError, ValueError) as exc:
            raise SpecInvalid(f"stored assertion set is not valid JSON: {exc}") from None
        if isinstance(doc, dict):
            require_supported(doc.get("schema_id"), doc.get("schema_version"))
            if doc.get("schema_id") != SCHEMA_ID_ASSERTION_SET or "assertions" not in doc:
                raise SpecInvalid("stored document is not an assertion-set document")
            doc = doc["assertions"]
        if not isinstance(doc, list):
            raise SpecInvalid("stored assertion set must be a JSON array")
        return doc


class InMemoryProgramRegistry:
    """Immutable ``{(id, version): sha256:<hex>}`` registry."""

    __slots__ = ("_hashes",)

    def __init__(self, hashes: Optional[Mapping[Tuple[str, str], str]] = None) -> None:
        object.__setattr__(self, "_hashes", dict(hashes or {}))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("InMemoryProgramRegistry is read-only")

    def registered_program_hash(self, program_id: str, version: str) -> Optional[str]:
        return self._hashes.get((program_id, version))


__all__ = ["MAX_SET_BYTES", "SpecStore", "ProgramRegistry", "InMemorySpecStore", "DirectorySpecStore",
           "InMemoryProgramRegistry"]

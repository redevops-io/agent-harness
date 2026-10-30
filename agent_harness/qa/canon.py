# SPDX-License-Identifier: AGPL-3.0-or-later
"""``canon-v1``: canonical JSON and SHA-256 for the qa/v1 payload subset.

An independent implementation of the published canonicalization rules (map keys sorted by Unicode code point, a null
map value is dropped so absent equals null, strings and keys are NFC, floats are refused, lists keep their order,
compact separators, UTF-8 without ``\\u`` escaping of non-ASCII). No source from any other package is used; the
frozen vectors in ``tests/qa/golden/qa_canon_vectors.json`` are the check.

Output: ``canon-v1:<64 lowercase hex>``. The contracts side writes the same digest as ``rcv1:<64 lowercase hex>``;
:func:`to_contract_ref` / :func:`from_contract_ref` map the prefix and :func:`same_digest` compares across it.

Fail-closed choices where the rules are silent (see HANDOFF.md):

* only exact ``str``, ``int``, ``bool`` and ``None`` scalars are accepted (subclasses such as ``IntEnum`` are refused);
* map keys must be ``str``; two keys that are equal after NFC are refused (no silent overwrite);
* a string that cannot be encoded as UTF-8 (a lone surrogate) is refused;
* nesting deeper than :data:`MAX_DEPTH` is refused (this also stops cycles);
* integers of any size are accepted: the rules refuse floats only.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Mapping
from typing import Any, List

from .errors import CanonicalizationRefused

ALGORITHM = "canon-v1"
PREFIX = "canon-v1:"
CONTRACT_PREFIX = "rcv1:"
MAX_DEPTH = 64

_HEX64 = re.compile(r"[0-9a-f]{64}")

_SHORT_ESCAPES = {'"': '\\"', "\\": "\\\\", "\n": "\\n", "\r": "\\r", "\t": "\\t", "\b": "\\b", "\f": "\\f"}


def _quote(text: str) -> str:
    """JSON string literal: escape ``"``, ``\\`` and C0 controls; leave '/', DEL and non-ASCII as they are."""
    out: List[str] = ['"']
    for ch in text:
        esc = _SHORT_ESCAPES.get(ch)
        if esc is not None:
            out.append(esc)
        elif ch < " ":
            out.append("\\u%04x" % ord(ch))
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def _nfc(text: str, what: str) -> str:
    try:
        text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise CanonicalizationRefused(f"{what} is not valid Unicode text (lone surrogate)") from exc
    return unicodedata.normalize("NFC", text)


def _emit(value: Any, depth: int, out: List[str]) -> None:
    if depth > MAX_DEPTH:
        raise CanonicalizationRefused(f"nesting deeper than {MAX_DEPTH} levels is refused")
    t = type(value)
    if value is None:
        out.append("null")
    elif t is bool:
        out.append("true" if value else "false")
    elif t is int:
        out.append(str(value))
    elif t is str:
        out.append(_quote(_nfc(value, "string")))
    elif isinstance(value, float):
        raise CanonicalizationRefused("a float has no canonical form; carry decimals as strings")
    elif isinstance(value, Mapping):
        items = {}
        for key, item in value.items():
            if type(key) is not str:
                raise CanonicalizationRefused(f"map keys must be strings, got {type(key).__name__}")
            nkey = _nfc(key, "map key")
            if nkey in items:
                raise CanonicalizationRefused(f"two map keys are equal after NFC normalization: {nkey!r}")
            items[nkey] = item
        out.append("{")
        first = True
        for nkey in sorted(items):                      # str ordering is by Unicode code point
            item = items[nkey]
            if item is None:                            # absent equals null: a null map value is dropped
                continue
            if not first:
                out.append(",")
            first = False
            out.append(_quote(nkey))
            out.append(":")
            _emit(item, depth + 1, out)
        out.append("}")
    elif isinstance(value, (list, tuple)):
        out.append("[")
        for i, item in enumerate(value):
            if i:
                out.append(",")
            _emit(item, depth + 1, out)                 # a null inside a list is an element and is kept
        out.append("]")
    else:
        raise CanonicalizationRefused(f"type {t.__name__} has no canonical form")


def canonical_text(value: Any) -> str:
    """The canonical JSON text of ``value``."""
    out: List[str] = []
    _emit(value, 0, out)
    return "".join(out)


def canonical_bytes(value: Any) -> bytes:
    """The canonical JSON of ``value`` as UTF-8 bytes (what is hashed)."""
    return canonical_text(value).encode("utf-8")


def digest_hex(value: Any) -> str:
    """SHA-256 of the canonical bytes as 64 lowercase hex digits."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def canon_hash(value: Any) -> str:
    """``canon-v1:<64 hex>`` for ``value``."""
    return PREFIX + digest_hex(value)


def is_canon_hash(text: Any) -> bool:
    return isinstance(text, str) and text.startswith(PREFIX) and _HEX64.fullmatch(text, len(PREFIX)) is not None


def is_contract_ref(text: Any) -> bool:
    return isinstance(text, str) and text.startswith(CONTRACT_PREFIX) and _HEX64.fullmatch(text, len(CONTRACT_PREFIX)) is not None


def to_contract_ref(canon_hash_text: str) -> str:
    """``canon-v1:<hex>`` -> ``rcv1:<hex>`` (the contracts spelling of the same digest)."""
    if not is_canon_hash(canon_hash_text):
        raise CanonicalizationRefused("not a canon-v1 hash")
    return CONTRACT_PREFIX + canon_hash_text[len(PREFIX):]


def from_contract_ref(contract_ref: str) -> str:
    """``rcv1:<hex>`` -> ``canon-v1:<hex>``."""
    if not is_contract_ref(contract_ref):
        raise CanonicalizationRefused("not an rcv1 hash")
    return PREFIX + contract_ref[len(CONTRACT_PREFIX):]


def digest_of(hash_text: str) -> str:
    """The 64 hex digits of a ``canon-v1:`` or ``rcv1:`` hash; anything else is refused."""
    if is_canon_hash(hash_text):
        return hash_text[len(PREFIX):]
    if is_contract_ref(hash_text):
        return hash_text[len(CONTRACT_PREFIX):]
    raise CanonicalizationRefused("not a canon-v1 or rcv1 hash (64 lowercase hex digits required)")


def same_digest(a: str, b: str) -> bool:
    """True when two ``canon-v1:``/``rcv1:`` hashes carry the same digest (malformed input is refused, not False)."""
    return digest_of(a) == digest_of(b)

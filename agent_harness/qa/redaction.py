# SPDX-License-Identifier: AGPL-3.0-or-later
"""A real redactor for the neutral ``Redactor`` port (plan H5), replacing the two-regex ``guardrails.redact`` idea.

:class:`PatternRedactor` is deterministic (an ordered rule library, no model, no clock, no randomness), idempotent
(redacting its own output changes nothing) and **fails closed**: input it cannot handle, bytes that are not text it can
decode, input over the size bound, a media type it has no rule for, and any unexpected internal error all raise the
typed ``RedactionFailed`` (code ``REDACTION_FAILED``). Callers store nothing in that case (see ``vault.py``); raw bytes
are never returned as a fallback.

What it removes (each rule has a stable id that is counted in the :class:`RedactionManifest`):

* a **deny-list of canary secrets** the operator supplies, matched literally and in their common encodings (percent,
  JSON, HTML, hex, base64 at three alignments); the policy id carries only the hash of each entry;
* private-key blocks (also an unterminated one, to the end of the text), credential-bearing header lines
  (``Authorization``, ``Cookie``, ``Set-Cookie``, API-key headers), userinfo in URLs, JWTs, provider key shapes
  (cloud, source host, chat, payments, model providers), secret-valued URL query parameters, ``name = value``
  assignments whose name says secret (password, token, api key, authorization, cookie ...), bearer and basic tokens,
  and high-entropy tokens by shape and Shannon entropy;
* optionally by policy (on by default: nothing personal is stored unless the operator turns it off): email addresses,
  phone numbers, payment card numbers (Luhn-checked).

For structured data (:meth:`PatternRedactor.redact_structure`, and JSON media types) the **key name** decides too: the
value of a key such as ``password``, ``token``, ``secret``, ``authorization``, ``cookie`` or ``api_key`` is replaced
as a whole, whatever it holds, and every other string is redacted as text. Counters whose key ends in ``tokens`` and
whose value is a number are left alone. Strings that are exactly a full-width digest (``sha256:``, ``rcv1:`` or
``canon-v1:`` plus 64 hex digits) are never touched: evidence references must stay intact.

Binary media (images, archives, a Playwright trace) is **not redacted and not claimed to be**: the audit's gate for
screenshots rests on synthetic data, so such an artifact fails closed unless the operator lists its media type as an
acknowledged exception (``RedactionConfig.acknowledged_unredacted_media``).

The ``policy_id`` of a redactor is the ``rcv1:`` digest of its whole rule set (rule ids and pattern sources, key
fragments, thresholds, flags, the hashed deny-list): the value for ``RunConfig.redaction_policy_id``, and it changes
when any rule does.

Standard library and neutral-module imports only.
"""
from __future__ import annotations

import base64
import hashlib
import html
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Pattern, Tuple
from urllib.parse import quote, quote_plus

from . import canon
from .errors import RedactionFailed, SpecInvalid

REDACTION_RULES_VERSION = "qa-redaction-rules/1"
POLICY_FORM = "qa-redaction-policy/1"
MANIFEST_FORM = "qa-redaction-manifest/1"
KEY_RULE = "key_name"
MAX_PASSES = 6
MIN_DENYLIST_CHARS = 8
MAX_DENYLIST_ENTRIES = 32
MAX_DENYLIST_CHARS = 4096
DEFAULT_MAX_INPUT_BYTES = 5 * 1024 * 1024
ENTROPY_MIN_MILLI = 3600            # bits per character times 1000, for a mixed-class token
ENTROPY_MIN_LENGTH = 24
HEX_MIN_LENGTH = 32

_DIGEST = re.compile(r"(?:sha256|rcv1|canon-v1):[0-9a-f]{64}")
_UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
_RULE_ID = re.compile(r"[a-z][a-z0-9_]{0,19}")


def token_for(rule_id: str) -> str:
    return "[REDACTED:" + rule_id + "]"


_SKIP = r"(?![\"']?\[REDACTED:)"      # a value that is already a redaction token is left alone (idempotence)

# ---------------------------------------------------------------------------------------------------------------
# the text rule library (order matters and is part of the policy id)
# ---------------------------------------------------------------------------------------------------------------

_HEADER_NAMES = (r"authorization|proxy-authorization|cookie|set-cookie|x-api-key|x-auth-token|x-csrf-token|"
                 r"x-xsrf-token|x-amz-security-token|x-goog-api-key|api-key|x-access-token")
_SECRET_NAME = (r"password|passwd|pwd|passphrase|secret|api[_-]?key|apikey|access[_-]?key|private[_-]?key|credential|"
                r"authorization|auth[_-]?token|access[_-]?token|refresh[_-]?token|id[_-]?token|client[_-]?secret|"
                r"session[_-]?id|sessionid|session[_-]?token|cookie|bearer|token|signature|csrf|xsrf")
_VALUE = (r"((?:(?:Bearer|Basic|Digest|Token|Negotiate|NTLM|AWS4-HMAC-SHA256)[ \t]+)?"
          r"(?:\"[^\"\r\n]*\"|'[^'\r\n]*'|[^\s\"',;&]+))")
_QUERY_NAMES = (r"access_token|id_token|refresh_token|token|api[_-]?key|apikey|key|secret|client_secret|password|"
                r"passwd|pwd|auth|authorization|sig|signature|code|session|sessionid|session_id|sid|jwt|"
                r"x-amz-signature|x-amz-credential|x-amz-security-token|x-goog-signature")

#: (rule id, pattern source, flags, which group is kept, mode); mode "value" keeps group 1 and replaces the rest
TEXT_RULES: Tuple[Tuple[str, str, int, str], ...] = (
    ("private_key", r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----[\s\S]*?"
                    r"(?:-----END (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----|\Z)", 0, "whole"),
    ("header_value", r"^([ \t]*(?:" + _HEADER_NAMES + r")[ \t]*+:[ \t]*+)" + _SKIP + r"[^\r\n]+", re.I | re.M, "keep1"),
    ("url_userinfo", r"(?<![A-Za-z0-9+.-])([a-z][a-z0-9+.-]*://)" + _SKIP + r"[^\s/?#@\"'<>]+@", re.I, "keep1_at"),
    ("jwt", r"eyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]*", 0, "whole"),
    ("aws_key_id", r"(?:AKIA|ASIA|AGPA|AIDA|AROA|ANPA|ANVA|AIPA|ABIA|ACCA)[A-Z0-9]{16}", 0, "whole"),
    ("google_key", r"AIza[0-9A-Za-z_-]{35}", 0, "whole"),
    ("github_token", r"(?:gh[pousr]_[A-Za-z0-9]{36,255}|github_pat_[A-Za-z0-9_]{22,255})", 0, "whole"),
    ("slack_token", r"xox[abprse]-[A-Za-z0-9-]{10,}", 0, "whole"),
    ("slack_webhook", r"https://hooks\.slack\.com/services/[A-Za-z0-9/_-]+", re.I, "whole"),
    ("stripe_key", r"(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,}", 0, "whole"),
    ("sk_style_key", r"(?<![A-Za-z0-9_])sk-(?:ant-|proj-|live-|test-)?[A-Za-z0-9_-]{20,}", 0, "whole"),
    ("npm_token", r"npm_[A-Za-z0-9]{36}", 0, "whole"),
    ("sendgrid_key", r"SG\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}", 0, "whole"),
    ("azure_secret", r"\b((?:AccountKey|SharedAccessKey|SharedAccessSignature)=)" + _SKIP + r"[^;\s\"']+", re.I,
     "keep1"),
    ("url_query", r"([?&;](?:" + _QUERY_NAMES + r")=)" + _SKIP + r"[^&#\s\"'<>]*", re.I, "keep1"),
    ("kv_secret", r"((?:" + _SECRET_NAME + r")[A-Za-z0-9_.-]{0,40}[\"']?[ \t]*+[:=][ \t]*+)" + _SKIP + _VALUE, re.I, "kv"),
    ("bearer", r"\b(bearer[ \t]+)" + _SKIP + r"[A-Za-z0-9._~+/=-]{8,}", re.I, "keep1"),
)

#: the optional PII rules (a rule is on unless its flag is turned off)
PII_RULES: Tuple[Tuple[str, str, int, str, str], ...] = (
    ("pii_email", r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}", 0, "whole",
     "pii_email"),
    ("pii_card", r"(?<![A-Za-z0-9])(?:[0-9][ -]?){12,18}[0-9](?![A-Za-z0-9])", 0, "luhn", "pii_card"),
    ("pii_phone", r"(?<![A-Za-z0-9_.+-])(?:\+\d{8,15}(?![0-9])|(?:\+\d{1,3}[ .-]?)?(?:\(\d{2,4}\)[ -]?|\d{2,4}[ -])"
                  r"\d{3,4}[ -]?\d{3,4}(?![A-Za-z0-9_-]))", 0, "whole", "pii_phone"),
)

#: key-name rule: the normalised key (lower case, letters and digits only) matches a fragment or an exact name
SENSITIVE_KEY_FRAGMENTS: Tuple[str, ...] = (
    "password", "passwd", "passphrase", "secret", "apikey", "accesskey", "privatekey", "credential", "authorization",
    "cookie", "bearer", "signature", "sessionid", "sessiontoken", "refreshtoken", "accesstoken", "idtoken", "authtoken",
    "csrf", "xsrf", "jwt", "token")
SENSITIVE_KEY_EXACT: Tuple[str, ...] = ("pwd", "auth", "sid", "session", "otp", "pin")


def normalise_key(key: str) -> str:
    return "".join(ch for ch in key.lower() if ch.isalnum())


# ---------------------------------------------------------------------------------------------------------------
# configuration, manifest
# ---------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True)
class RedactionConfig:
    """Operator configuration. The defaults are the most protective setting."""

    denylist: Tuple[str, ...] = ()                       # canary or known secrets: literal and encoded forms
    pii_email: bool = True
    pii_phone: bool = True
    pii_card: bool = True
    max_input_bytes: int = DEFAULT_MAX_INPUT_BYTES
    max_nodes: int = 200_000
    max_depth: int = 64
    acknowledged_unredacted_media: Tuple[str, ...] = ()  # media types the operator accepts WITHOUT redaction
    extra_sensitive_keys: Tuple[str, ...] = ()           # more normalised key fragments

    def __post_init__(self) -> None:
        for name in ("pii_email", "pii_phone", "pii_card"):
            if type(getattr(self, name)) is not bool:
                raise SpecInvalid(f"RedactionConfig.{name}: expected a boolean")
        for name, low, high in (("max_input_bytes", 1, 1 << 30), ("max_nodes", 1, 10_000_000), ("max_depth", 1, 200)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise SpecInvalid(f"RedactionConfig.{name}: expected an integer between {low} and {high}")
        deny = tuple(self.denylist)
        if len(deny) > MAX_DENYLIST_ENTRIES or any(type(d) is not str or not MIN_DENYLIST_CHARS <= len(d) <= MAX_DENYLIST_CHARS
                                                    for d in deny):
            raise SpecInvalid(f"RedactionConfig.denylist: up to {MAX_DENYLIST_ENTRIES} strings of "
                              f"{MIN_DENYLIST_CHARS} to {MAX_DENYLIST_CHARS} characters")
        raw_media = tuple(self.acknowledged_unredacted_media)
        if any(type(m) is not str or "/" not in m or ";" in m for m in raw_media):
            raise SpecInvalid("RedactionConfig.acknowledged_unredacted_media: expected bare media types (type/subtype)")
        media = tuple(sorted({m.strip().lower() for m in raw_media}))
        raw_keys = tuple(self.extra_sensitive_keys)
        if any(type(k) is not str for k in raw_keys):
            raise SpecInvalid("RedactionConfig.extra_sensitive_keys: expected strings")
        keys = tuple(sorted({normalise_key(k) for k in raw_keys}))
        if any(not k for k in keys):
            raise SpecInvalid("RedactionConfig.extra_sensitive_keys: expected non-empty names")
        object.__setattr__(self, "denylist", tuple(sorted({unicodedata.normalize("NFC", d) for d in deny})))
        object.__setattr__(self, "acknowledged_unredacted_media", media)
        object.__setattr__(self, "extra_sensitive_keys", keys)


@dataclass(frozen=True)
class RedactionManifest:
    """What one redaction did: counts per rule, the rule-set hash and the sizes. No content, and no hash of the raw
    input (which would let a low-entropy secret be confirmed by guessing)."""

    rule_set_hash: str
    counts: Tuple[Tuple[str, int], ...]
    input_bytes: int
    output_bytes: int
    media_type: str
    passthrough: bool = False       # the media type was an acknowledged exception: nothing was redacted

    @property
    def total(self) -> int:
        return sum(n for _, n in self.counts)

    def to_json(self) -> Dict[str, Any]:
        return {"counts": {k: v for k, v in self.counts}, "form": MANIFEST_FORM, "input_bytes": self.input_bytes,
                "media_type": self.media_type, "output_bytes": self.output_bytes, "passthrough": self.passthrough,
                "rule_set_hash": self.rule_set_hash, "total": self.total}

    @property
    def manifest_hash(self) -> str:
        return canon.canon_hash(self.to_json())


class RedactionTally:
    """Counts per rule across many redactions (a run's events). ``to_json`` is the run's redaction manifest."""

    def __init__(self, rule_set_hash: str) -> None:
        self.rule_set_hash = rule_set_hash
        self._counts: Dict[str, int] = {}
        self.items_redacted = 0
        self.items_total = 0

    def add(self, counts: Mapping[str, int]) -> None:
        self.items_total += 1
        if counts:
            self.items_redacted += 1
        for key, n in counts.items():
            self._counts[key] = self._counts.get(key, 0) + n

    def to_json(self) -> Dict[str, Any]:
        return {"counts": dict(sorted(self._counts.items())), "form": MANIFEST_FORM, "items_redacted": self.items_redacted,
                "items_total": self.items_total, "rule_set_hash": self.rule_set_hash,
                "total": sum(self._counts.values())}


# ---------------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------------

def _luhn(digits: str) -> bool:
    total, flip = 0, False
    for ch in reversed(digits):
        d = ord(ch) - 48
        if flip:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        flip = not flip
    return total % 10 == 0


def _entropy_milli(token: str) -> int:
    counts: Dict[str, int] = {}
    for ch in token:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(token)
    h = -sum(c / n * math.log2(c / n) for c in counts.values())
    return int(h * 1000)


def _b64_variants(data: bytes) -> List[str]:
    out: List[str] = []
    for off in range(3):
        enc = base64.b64encode(b"\x00" * off + data).decode("ascii")
        skip = off * 8 // 6 + (1 if off * 8 % 6 else 0)
        body = enc.rstrip("=")
        if (off + len(data)) % 3:
            body = body[:-1]                    # the last character depends on bytes that are not part of the secret
        body = body[skip:]
        if len(body) >= 6:
            out.extend((body, body.replace("+", "-").replace("/", "_")))
    full = base64.b64encode(data).decode("ascii")               # the whole, padded encoding of a standalone secret
    for form in (full, full.rstrip("=")):
        out.extend((form, form.replace("+", "-").replace("/", "_")))
    return out


def denylist_variants(secret: str) -> List[str]:
    """The forms a canary may take once it has been encoded on its way into evidence."""
    data = secret.encode("utf-8")
    forms = {secret, quote(secret, safe=""), quote_plus(secret), json.dumps(secret, ensure_ascii=True)[1:-1],
             json.dumps(secret, ensure_ascii=False)[1:-1], html.escape(secret, quote=True),
             "".join("&#%d;" % ord(c) for c in secret), "".join("&#x%x;" % ord(c) for c in secret), data.hex(),
             "".join("\\x%02x" % b for b in data), secret.replace("/", "\\/")}
    forms.update(_b64_variants(data))
    return sorted((f for f in forms if len(f) >= MIN_DENYLIST_CHARS - 2), key=lambda f: (-len(f), f))


_TEXT_MEDIA_EXACT = frozenset({"application/json", "application/xml", "application/javascript", "application/x-ndjson",
                               "application/x-www-form-urlencoded", "application/yaml", "application/x-yaml",
                               "application/xhtml+xml", "application/ecmascript", "application/sql",
                               "application/x-sh", "application/toml"})
_OK_CHARSETS = frozenset({"utf-8", "utf8", "us-ascii", "ascii"})


def _split_media_type(media_type: Any) -> Tuple[str, Optional[str]]:
    if type(media_type) is not str or not media_type.strip():
        raise RedactionFailed("redaction needs a media type", reason="UNSUPPORTED_MEDIA_TYPE")
    base, _, params = media_type.partition(";")
    charset = None
    for part in params.split(";"):
        key, _, value = part.partition("=")
        if key.strip().lower() == "charset":
            charset = value.strip().strip('"').lower()
    return base.strip().lower(), charset


def is_text_media_type(base: str) -> bool:
    return base.startswith("text/") or base in _TEXT_MEDIA_EXACT or base.endswith("+json") or base.endswith("+xml")


def is_json_media_type(base: str) -> bool:
    return base == "application/json" or base.endswith("+json")


# ---------------------------------------------------------------------------------------------------------------
# the redactor
# ---------------------------------------------------------------------------------------------------------------

def _fail(reason: str, message: str) -> RedactionFailed:
    return RedactionFailed(message, reason=reason)


class PatternRedactor:
    """The neutral ``Redactor`` (``policy_id`` and ``redact(data, media_type) -> bytes``) plus manifests and structure."""

    def __init__(self, config: Optional[RedactionConfig] = None) -> None:
        self._config = config if config is not None else RedactionConfig()
        if type(self._config) is not RedactionConfig:
            raise SpecInvalid("PatternRedactor: config must be a RedactionConfig")
        self._rules: List[Tuple[str, Pattern[str], str]] = []
        self._rule_docs: List[Dict[str, Any]] = []
        for rule_id, source, flags, mode in TEXT_RULES:
            self._add(rule_id, source, flags, mode)
        for rule_id, source, flags, mode, flag in PII_RULES:
            if getattr(self._config, flag):
                self._add(rule_id, source, flags, mode)
        self._deny_variants = self._build_denylist()
        keys = SENSITIVE_KEY_FRAGMENTS + self._config.extra_sensitive_keys
        self._key_fragments = keys
        self._key_exact = frozenset(SENSITIVE_KEY_EXACT)
        doc = {"denylist": [hashlib.sha256(d.encode("utf-8")).hexdigest() for d in self._config.denylist],
               "entropy": {"hex_min_length": HEX_MIN_LENGTH, "min_length": ENTROPY_MIN_LENGTH,
                           "min_milli": ENTROPY_MIN_MILLI},
               "form": POLICY_FORM, "key_exact": sorted(self._key_exact), "key_fragments": list(keys),
               "max_passes": MAX_PASSES,
               "limits": {"max_depth": self._config.max_depth, "max_input_bytes": self._config.max_input_bytes,
                          "max_nodes": self._config.max_nodes},
               "rules": self._rule_docs, "rules_version": REDACTION_RULES_VERSION,
               "unredacted_media": list(self._config.acknowledged_unredacted_media)}
        self._policy_id = canon.to_contract_ref(canon.canon_hash(doc))

    @property
    def policy_id(self) -> str:
        """``rcv1:<64 hex>`` of the rule set: the value for ``RunConfig.redaction_policy_id``."""
        return self._policy_id

    @property
    def config(self) -> RedactionConfig:
        return self._config

    def _add(self, rule_id: str, source: str, flags: int, mode: str) -> None:
        assert _RULE_ID.fullmatch(rule_id), rule_id
        self._rules.append((rule_id, re.compile(source, flags), mode))
        self._rule_docs.append({"flags": int(flags), "id": rule_id, "mode": mode, "source": source})

    def _build_denylist(self) -> Tuple[str, ...]:
        """Every literal form of every deny-list entry (case variants included), longest first. Matching is by plain
        substring search, so the cost is linear in the input for each form."""
        forms = set()
        for secret in self._config.denylist:
            for form in denylist_variants(secret):
                forms.update((form, form.lower(), form.upper()))
        ids = {rule_id for rule_id, _, _, _ in TEXT_RULES} | {r[0] for r in PII_RULES} | {
            KEY_RULE, "entropy_token", "hex_token", "denylist"}
        tokens = " ".join(token_for(i) for i in sorted(ids)).lower()
        if any(f.lower() in tokens for f in forms):
            raise SpecInvalid("RedactionConfig.denylist: an entry (or one of its encodings) occurs inside a redaction token")
        return tuple(sorted(forms, key=lambda f: (-len(f), f)))

    # -- text ----------------------------------------------------------------------------------------------
    def redact_text(self, text: str) -> Tuple[str, Dict[str, int]]:
        """Redact one string. Returns the text and the counts per rule. Strings that are exactly a full-width digest
        are returned unchanged."""
        counts: Dict[str, int] = {}
        return self._text(text, counts), counts

    def _text(self, text: str, counts: Dict[str, int]) -> str:
        """Redact to a fixed point: another pass over the result changes nothing, by construction (a rule's boundary can
        depend on text a neighbouring rule replaced, so one pass is not always enough). A text that does not settle in
        ``MAX_PASSES`` passes is refused."""
        if _DIGEST.fullmatch(text):
            return text
        for _ in range(MAX_PASSES):
            result = self._pass(text, counts)
            if result == text:
                return result
            text = result
        raise _fail("UNSTABLE", "the redaction did not settle")

    def _pass(self, text: str, counts: Dict[str, int]) -> str:
        for variant in self._deny_variants:
            n = text.count(variant)
            if n:
                counts["denylist"] = counts.get("denylist", 0) + n
                text = text.replace(variant, token_for("denylist"))
        for rule_id, pattern, mode in self._rules:
            text = self._apply(rule_id, pattern, mode, text, counts)
        return self._entropy(text, counts)

    @staticmethod
    def _apply(rule_id: str, pattern: Pattern[str], mode: str, text: str, counts: Dict[str, int]) -> str:
        token = token_for(rule_id)

        def hit(n: int = 1) -> None:
            counts[rule_id] = counts.get(rule_id, 0) + n

        if mode == "whole":
            def repl_whole(m: "re.Match[str]") -> str:
                hit()
                return token
            return pattern.sub(repl_whole, text)
        if mode == "keep1":
            def repl_keep(m: "re.Match[str]") -> str:
                hit()
                return m.group(1) + token
            return pattern.sub(repl_keep, text)
        if mode == "kv":
            def repl_kv(m: "re.Match[str]") -> str:
                hit()
                value = m.group(2)
                q = value[0] if value[0] in "\"'" else ""
                return m.group(1) + q + token + q
            return pattern.sub(repl_kv, text)
        if mode == "keep1_at":
            def repl_at(m: "re.Match[str]") -> str:
                hit()
                return m.group(1) + token + "@"
            return pattern.sub(repl_at, text)
        if mode == "luhn":
            def repl_luhn(m: "re.Match[str]") -> str:
                digits = "".join(ch for ch in m.group(0) if ch.isdigit())
                if 13 <= len(digits) <= 19 and _luhn(digits):
                    hit()
                    return token
                return m.group(0)
            return pattern.sub(repl_luhn, text)
        raise _fail("INTERNAL", "unknown rule mode")

    _CANDIDATE = re.compile(
        r"(?P<digest>(?<![A-Za-z0-9+/_-])(?:sha256|rcv1|canon-v1):[0-9a-f]{64}(?![A-Za-z0-9+/_=-]))"
        r"|(?P<token>(?<![A-Za-z0-9+/_-])[A-Za-z0-9+/_-]{%d,}={0,2}(?![A-Za-z0-9+/_=-]))" % ENTROPY_MIN_LENGTH)

    def _entropy(self, text: str, counts: Dict[str, int]) -> str:
        def repl(m: "re.Match[str]") -> str:
            tok = m.group("token")
            if tok is None:                                     # a full-width digest reference: left exactly as it is
                return m.group(0)
            rule = self._entropy_rule(tok)
            if rule is None:
                return tok
            counts[rule] = counts.get(rule, 0) + 1
            return token_for(rule)
        return self._CANDIDATE.sub(repl, text)

    @classmethod
    def _entropy_rule(cls, tok: str) -> Optional[str]:
        if tok.count("/") and "+" not in tok and "=" not in tok:       # path-like: judge each segment on its own
            parts = [p for p in tok.split("/")]
            if any(cls._entropy_rule(p) is not None for p in parts if len(p) >= ENTROPY_MIN_LENGTH):
                return "entropy_token"
            return None
        if len(tok) >= HEX_MIN_LENGTH and all(c in "0123456789abcdefABCDEF" for c in tok):
            return "hex_token"
        if _UUID.fullmatch(tok):
            return None
        if any(c.isupper() for c in tok) and any(c.islower() for c in tok) and any(c.isdigit() for c in tok) \
                and _entropy_milli(tok) >= ENTROPY_MIN_MILLI:
            return "entropy_token"
        return None

    # -- structure -----------------------------------------------------------------------------------------
    def _key_sensitive(self, key: str) -> bool:
        norm = normalise_key(key)
        return bool(norm) and (norm in self._key_exact or any(f in norm for f in self._key_fragments))

    def redact_structure(self, value: Any, *, exempt_keys: Iterable[str] = ()) -> Tuple[Any, Dict[str, int]]:
        """Redact a JSON-shaped value: dictionaries, lists, strings, numbers, booleans and ``None``. The key name
        decides first; ``exempt_keys`` are normalised key names that are structure, not secrets (for example a record's
        ``secret_ref``). Tuples become lists. Raises ``RedactionFailed`` for anything else, for too much depth or size,
        and for two keys that become equal after redaction."""
        exempt = frozenset(normalise_key(k) for k in exempt_keys)
        counts: Dict[str, int] = {}
        nodes = [0]
        cfg = self._config

        def walk(v: Any, depth: int) -> Any:
            nodes[0] += 1
            if nodes[0] > cfg.max_nodes:
                raise _fail("TOO_MANY_NODES", "the structure has too many nodes")
            if depth > cfg.max_depth:
                raise _fail("TOO_DEEP", "the structure is nested too deeply")
            t = type(v)
            if t is str:
                return self._text(v, counts)
            if v is None or t is bool or t is int:
                return v
            if t is float:
                if not math.isfinite(v):
                    raise _fail("UNSUPPORTED_TYPE", "a non-finite number has no JSON form")
                return v
            if isinstance(v, Mapping):
                out: Dict[str, Any] = {}
                for key, item in v.items():
                    if type(key) is not str:
                        raise _fail("UNSUPPORTED_TYPE", "a key is not a string")
                    shown = self._text(key, counts)
                    if shown in out:
                        raise _fail("KEY_COLLISION", "two keys are equal after redaction")
                    norm = normalise_key(key)
                    if norm not in exempt and self._key_sensitive(key):
                        if norm.endswith("tokens") and type(item) in (int, bool):
                            out[shown] = item                           # a counter, not a secret
                        elif item is None:
                            out[shown] = None
                        else:
                            counts[KEY_RULE] = counts.get(KEY_RULE, 0) + 1
                            out[shown] = token_for(KEY_RULE)
                    else:
                        out[shown] = walk(item, depth + 1)
                return out
            if isinstance(v, (list, tuple)):
                return [walk(item, depth + 1) for item in v]
            raise _fail("UNSUPPORTED_TYPE", "a value has no JSON form")

        try:
            return walk(value, 0), counts
        except RedactionFailed:
            raise
        except Exception as exc:                                        # fail closed on anything unexpected
            raise _fail("INTERNAL", "redaction failed (%s)" % type(exc).__name__) from exc

    # -- bytes (the port) ----------------------------------------------------------------------------------
    def redact_with_manifest(self, data: Any, media_type: Any) -> Tuple[bytes, RedactionManifest]:
        """Redact bytes and report what was done. Fails closed: ``RedactionFailed`` for input that is not bytes, is
        over ``max_input_bytes``, is not decodable text of an accepted charset, has a media type without a rule (unless
        acknowledged), or on any internal error. Nothing raw is ever returned."""
        try:
            return self._redact_bytes(data, media_type)
        except RedactionFailed:
            raise
        except Exception as exc:
            raise _fail("INTERNAL", "redaction failed (%s)" % type(exc).__name__) from exc

    def redact(self, data: bytes, media_type: str) -> bytes:
        return self.redact_with_manifest(data, media_type)[0]

    def _redact_bytes(self, data: Any, media_type: Any) -> Tuple[bytes, RedactionManifest]:
        if type(data) not in (bytes, bytearray):
            raise _fail("INPUT_NOT_BYTES", "redaction takes bytes")
        if len(data) > self._config.max_input_bytes:
            raise _fail("INPUT_TOO_LARGE", "the input is over the size bound")
        base, charset = _split_media_type(media_type)
        shown_type = base
        if not is_text_media_type(base):
            if base in self._config.acknowledged_unredacted_media:
                return bytes(data), RedactionManifest(rule_set_hash=self._policy_id, counts=(), input_bytes=len(data),
                                                      output_bytes=len(data), media_type=shown_type, passthrough=True)
            raise _fail("UNSUPPORTED_MEDIA_TYPE", "this media type cannot be redacted")
        if charset is not None and charset not in _OK_CHARSETS:
            raise _fail("UNSUPPORTED_CHARSET", "only UTF-8 text can be redacted")
        try:
            text = bytes(data).decode("utf-8")
        except UnicodeDecodeError:
            raise _fail("NOT_UTF8", "the input is not valid UTF-8") from None
        counts: Dict[str, int] = {}
        if is_json_media_type(base):
            parsed = self._parse_json(text)
            if parsed is _NOT_JSON:
                out_text = self._text(text, counts)                      # malformed JSON: redact it as text
            else:
                value, counts = self.redact_structure(parsed)
                out_text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        else:
            out_text = self._text(text, counts)
        try:
            out = out_text.encode("utf-8")
        except UnicodeEncodeError:
            raise _fail("NOT_ENCODABLE", "the output is not encodable as UTF-8") from None
        manifest = RedactionManifest(rule_set_hash=self._policy_id, counts=tuple(sorted(counts.items())),
                                     input_bytes=len(data), output_bytes=len(out), media_type=shown_type)
        return out, manifest

    @staticmethod
    def _parse_json(text: str) -> Any:
        def refuse(_: str) -> Any:
            raise ValueError("non-finite number")
        try:
            return json.loads(text, parse_constant=refuse)
        except (ValueError, RecursionError):
            return _NOT_JSON


_NOT_JSON = object()

__all__ = ["REDACTION_RULES_VERSION", "POLICY_FORM", "MANIFEST_FORM", "KEY_RULE", "TEXT_RULES", "PII_RULES",
           "SENSITIVE_KEY_FRAGMENTS", "SENSITIVE_KEY_EXACT", "RedactionConfig", "RedactionManifest", "RedactionTally",
           "PatternRedactor", "token_for", "normalise_key", "denylist_variants", "is_text_media_type",
           "is_json_media_type"]

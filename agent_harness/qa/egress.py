# SPDX-License-Identifier: AGPL-3.0-or-later
"""The QA egress policy (plan H5): pure, deterministic, default-deny, with an injected resolver.

This module decides; it never connects. The point that enforces the decision (a worker or browser egress gateway, the
Chromium loopback proxy-bypass test) is ``redevops-qa`` PR A2 and calls into this logic. Nothing here opens a socket or
resolves a name: name resolution is an injected :class:`Resolver`, and there is deliberately no default that reaches
the network (:class:`NoResolver` refuses, so a guard built without one can only decide for address literals).

What a decision rests on (ADR Addendum A.8, audit threat T3, decision D10):

* **The allow-list is operator data**, never page content and never the spec (:class:`EgressPolicy`, built from
  :class:`AllowEntry` values). With no entry every request is denied. An entry names an exact ``scheme``, ``host`` and
  ``port``; a wildcard is only a subdomain wildcard of a named public domain (never the apex, never a bare suffix);
  an entry that admits a private, loopback or other non-public target (``private_ok``) must name the exact networks
  (single addresses or CIDRs) the target may resolve to, and the ports.
* **Metadata and link-local addresses are refused always**, whatever the allow-list says: ``169.254.0.0/16``
  (``169.254.169.254``), ``fe80::/10``, ``fd00:ec2::/32`` (``fd00:ec2::254``), ``fd20:ce::/32``, ``100.100.100.200``,
  ``168.63.129.16`` and the metadata and internal-suffix names, also when they are hidden inside an IPv4-mapped,
  IPv4-compatible, NAT64, 6to4 or Teredo address. The check runs before the allow-list is read.
* **Resolve, then pin.** The resolver is called once per decision; the decision is made on *all* the addresses it
  returned (one bad answer among good ones denies the whole set, which is the DNS-rebinding and split-answer defence);
  an allowed decision carries ``pinned_address`` and a connector must connect to exactly that address
  (:meth:`EgressDecision.connect_target`, :meth:`EgressGuard.verify_peer`). There is no second resolution.
* **Every redirect hop is decided again** from scratch (:func:`decide_chain`, :meth:`EgressGuard.check_redirect`), a
  relative ``Location`` is resolved against the previous hop, an ``https`` to ``http`` downgrade is denied, and the
  number of redirects is capped.
* **The same rules for every kind of request**: navigation, subresources, WebSockets (``ws`` and ``wss`` are judged
  as the same-origin ``http`` and ``https`` entry) and redirects. Only ``http``, ``https``, ``ws`` and ``wss`` can ever
  be admitted; every other scheme is refused and cannot be listed.
* **Odd spellings do not slip through.** The host is normalised before it is judged: userinfo is refused, percent
  encoding is decoded once (a second level is refused), NFKC and ideographic dots are folded, IDNA is applied,
  a single trailing dot is dropped, and every numeric IPv4 spelling (decimal, octal, hex, short forms) and bracketed
  IPv6 is parsed to the address the browser would reach. The path, query and fragment never take part in a decision
  and never appear in a decision record (denied hosts are counted by host only).
* **Every decision is a record.** :class:`EgressDecision` is plain data with a ``canon-v1`` hash; the guard keeps
  every decision, allow or deny, until the runner drains them into the evidence stream.

Interpretation choices are listed in the H5 handoff (HANDOFF.md). Standard library and neutral-module imports only.
"""
from __future__ import annotations

import enum
import ipaddress
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple, Union, \
    runtime_checkable
from urllib.parse import unquote_to_bytes

from . import canon
from .errors import SpecInvalid

EGRESS_RULES_VERSION = "qa-egress-rules/1"      # bump when a rule changes: it is part of the policy id
DECISION_FORM = "qa-egress-decision/1"
POLICY_FORM = "qa-egress-policy/1"

MAX_URL_CHARS = 2048
MAX_HOST_CHARS = 253
MAX_REDIRECTS_LIMIT = 20
DEFAULT_MAX_REDIRECTS = 5
DEFAULT_MAX_ADDRESSES = 16
MAX_PENDING_DECISIONS = 4096
MIN_PRIVATE_PREFIX_V4 = 16        # an operator entry broader than this is refused when the policy is built
MIN_PRIVATE_PREFIX_V6 = 48

SCHEMES = ("http", "https", "ws", "wss")
DEFAULT_PORTS = {"http": 80, "https": 443, "ws": 80, "wss": 443}
_SAME_ORIGIN_ENTRY = {"ws": "http", "wss": "https"}
_SECURE = frozenset({"https", "wss"})


def _downgrade(prev_scheme: str, scheme: str) -> bool:
    return prev_scheme in _SECURE and scheme not in _SECURE
#: the one non-network URL a browser starts on; it is not a request and never reaches a decision as a target
ABOUT_BLANK = "about:blank"


# ---------------------------------------------------------------------------------------------------------------
# reason codes
# ---------------------------------------------------------------------------------------------------------------

class Reason(enum.StrEnum):
    ALLOWED_PUBLIC = "ALLOWED_PUBLIC"
    ALLOWED_PRIVATE_OPERATOR = "ALLOWED_PRIVATE_OPERATOR"
    STATIC_ONLY = "STATIC_ONLY"                          # listed host, addresses not checked (no resolution)
    DENY_NOT_LISTED = "DENY_NOT_LISTED"                  # default deny
    DENY_MALFORMED_URL = "DENY_MALFORMED_URL"
    DENY_SCHEME = "DENY_SCHEME"
    DENY_USERINFO = "DENY_USERINFO"
    DENY_HOST_SYNTAX = "DENY_HOST_SYNTAX"
    DENY_PORT = "DENY_PORT"
    DENY_METADATA_ADDRESS = "DENY_METADATA_ADDRESS"      # always, even when listed
    DENY_METADATA_NAME = "DENY_METADATA_NAME"            # always, even when listed
    DENY_INTERNAL_NAME = "DENY_INTERNAL_NAME"            # .internal, .local, ... always, even when listed
    DENY_NO_ADDRESSES = "DENY_NO_ADDRESSES"
    DENY_TOO_MANY_ADDRESSES = "DENY_TOO_MANY_ADDRESSES"
    DENY_RESOLVER_ERROR = "DENY_RESOLVER_ERROR"
    DENY_RESOLVER_BAD_ANSWER = "DENY_RESOLVER_BAD_ANSWER"
    DENY_ADDRESS_NOT_PUBLIC = "DENY_ADDRESS_NOT_PUBLIC"  # a public entry whose host resolves to a non-public address
    DENY_ADDRESS_MAPPED = "DENY_ADDRESS_MAPPED"          # IPv4-mapped / compatible IPv6
    DENY_ADDRESS_TRANSITION = "DENY_ADDRESS_TRANSITION"  # NAT64, 6to4, Teredo
    DENY_ADDRESS_OUTSIDE_ENTRY = "DENY_ADDRESS_OUTSIDE_ENTRY"
    DENY_REDIRECT_LIMIT = "DENY_REDIRECT_LIMIT"
    DENY_REDIRECT_LOCATION = "DENY_REDIRECT_LOCATION"
    DENY_SCHEME_DOWNGRADE = "DENY_SCHEME_DOWNGRADE"
    DENY_PEER_MISMATCH = "DENY_PEER_MISMATCH"            # the connection did not go to the pinned address (rebinding)


class EgressVerdict(enum.StrEnum):
    ALLOW = "ALLOW"
    DENY = "DENY"


class AddressClass(enum.StrEnum):
    PUBLIC = "PUBLIC"
    METADATA = "METADATA"
    LINK_LOCAL = "LINK_LOCAL"
    LOOPBACK = "LOOPBACK"
    PRIVATE = "PRIVATE"            # RFC 1918
    CGNAT = "CGNAT"
    ULA = "ULA"
    MULTICAST = "MULTICAST"
    UNSPECIFIED = "UNSPECIFIED"
    RESERVED = "RESERVED"          # documentation, benchmarking, protocol assignments, class E, broadcast, ...
    MAPPED = "MAPPED"
    TRANSITION = "TRANSITION"


class Refused(Exception):
    """A target that cannot be admitted, with the reason code (raised by :func:`parse_target` and
    :func:`resolve_location`; the decision functions catch it and return a ``DENY`` record)."""

    def __init__(self, reason: Reason) -> None:
        super().__init__(reason.value)
        self.reason = reason


# ---------------------------------------------------------------------------------------------------------------
# address classification
# ---------------------------------------------------------------------------------------------------------------

def _nets(*texts: str) -> Tuple[Any, ...]:
    return tuple(ipaddress.ip_network(t) for t in texts)


#: refused always, even when an operator lists them (metadata services and link-local space)
_METADATA_NETS = _nets("169.254.0.0/16", "fe80::/10", "fd00:ec2::/32", "fd20:ce::/32", "100.100.100.200/32",
                       "168.63.129.16/32", "192.0.0.192/32")

_V4_CLASSES: Tuple[Tuple[Any, AddressClass], ...] = tuple((ipaddress.ip_network(n), c) for n, c in (
    ("0.0.0.0/8", AddressClass.UNSPECIFIED), ("10.0.0.0/8", AddressClass.PRIVATE),
    ("100.64.0.0/10", AddressClass.CGNAT), ("127.0.0.0/8", AddressClass.LOOPBACK),
    ("169.254.0.0/16", AddressClass.LINK_LOCAL), ("172.16.0.0/12", AddressClass.PRIVATE),
    ("192.0.0.0/24", AddressClass.RESERVED), ("192.0.2.0/24", AddressClass.RESERVED),
    ("192.88.99.0/24", AddressClass.RESERVED), ("192.168.0.0/16", AddressClass.PRIVATE),
    ("198.18.0.0/15", AddressClass.RESERVED), ("198.51.100.0/24", AddressClass.RESERVED),
    ("203.0.113.0/24", AddressClass.RESERVED), ("224.0.0.0/4", AddressClass.MULTICAST),
    ("240.0.0.0/4", AddressClass.RESERVED)))

_V6_CLASSES: Tuple[Tuple[Any, AddressClass], ...] = tuple((ipaddress.ip_network(n), c) for n, c in (
    ("::/128", AddressClass.UNSPECIFIED), ("::1/128", AddressClass.LOOPBACK),
    ("fc00::/7", AddressClass.ULA), ("fe80::/10", AddressClass.LINK_LOCAL),
    ("fec0::/10", AddressClass.RESERVED), ("ff00::/8", AddressClass.MULTICAST),
    ("2001:db8::/32", AddressClass.RESERVED), ("2001::/23", AddressClass.RESERVED),
    ("3fff::/20", AddressClass.RESERVED)))

_GLOBAL_UNICAST_V6 = ipaddress.ip_network("2000::/3")
_V6_MAPPED = ipaddress.ip_network("::ffff:0:0/96")
_V6_NAT64 = ipaddress.ip_network("64:ff9b::/96")
_V6_NAT64_LOCAL = ipaddress.ip_network("64:ff9b:1::/48")
_V6_6TO4 = ipaddress.ip_network("2002::/16")
_V6_TEREDO = ipaddress.ip_network("2001::/32")
_V6_COMPAT = ipaddress.ip_network("::/96")


def address_text(addr: Any) -> str:
    """Canonical text of an address that does not depend on the Python version: dotted decimal for IPv4, the full
    eight-group form for IPv6."""
    if addr.version == 4:
        return str(addr)
    n = int(addr)                       # not ``exploded``: Python 3.13 prints a mapped address with a dotted tail
    return ":".join("%04x" % ((n >> shift) & 0xFFFF) for shift in range(112, -1, -16))


def _embedded_v4(addr: Any) -> Optional[Any]:
    """The IPv4 address hidden inside a mapped, compatible, NAT64, 6to4 or Teredo address (``None`` if none)."""
    if addr.version != 6:
        return None
    n = int(addr)
    if addr in _V6_MAPPED:
        return ipaddress.IPv4Address(n & 0xFFFFFFFF)
    if addr in _V6_NAT64 or addr in _V6_NAT64_LOCAL:
        return ipaddress.IPv4Address(n & 0xFFFFFFFF)
    if addr in _V6_6TO4:
        return ipaddress.IPv4Address((n >> 80) & 0xFFFFFFFF)
    if addr in _V6_TEREDO:
        return ipaddress.IPv4Address((n & 0xFFFFFFFF) ^ 0xFFFFFFFF)
    if addr in _V6_COMPAT and n > 1:
        return ipaddress.IPv4Address(n & 0xFFFFFFFF)
    return None


def _in_metadata(addr: Any) -> bool:
    return any(addr in net for net in _METADATA_NETS if net.version == addr.version)


def is_metadata_address(addr: Any) -> bool:
    """True for a metadata or link-local address, including one hidden in a transition or mapped form."""
    if _in_metadata(addr):
        return True
    inner = _embedded_v4(addr)
    return inner is not None and _in_metadata(inner)


def classify_address(addr: Any) -> AddressClass:
    """The class of an address. Anything that is not provably a public unicast address is not ``PUBLIC``."""
    if is_metadata_address(addr):
        return AddressClass.METADATA
    if addr.version == 4:
        for net, cls in _V4_CLASSES:
            if addr in net:
                return cls
        return AddressClass.PUBLIC
    if addr in _V6_MAPPED or addr in _V6_COMPAT and int(addr) > 1:
        return AddressClass.MAPPED
    if addr in _V6_NAT64 or addr in _V6_NAT64_LOCAL or addr in _V6_6TO4 or addr in _V6_TEREDO:
        return AddressClass.TRANSITION
    for net, cls in _V6_CLASSES:
        if addr in net:
            return cls
    if addr not in _GLOBAL_UNICAST_V6:
        return AddressClass.RESERVED
    return AddressClass.PUBLIC


# ---------------------------------------------------------------------------------------------------------------
# names
# ---------------------------------------------------------------------------------------------------------------

#: metadata service names, refused always
METADATA_NAMES = frozenset({"metadata", "metadata.google.internal", "metadata.goog", "instance-data",
                            "instance-data.ec2.internal"})
#: names that only ever mean "inside": refused always (D10 rule 5: ``.internal`` and ``.local`` even if listed)
INTERNAL_SUFFIXES = (".internal", ".local", ".localhost", ".localdomain", ".home.arpa", ".intranet", ".lan", ".corp",
                     ".private")

_LABEL = re.compile(r"[a-z0-9_](?:[a-z0-9_-]{0,61}[a-z0-9_])?")
_NUMERIC_LABEL = re.compile(r"0[xX][0-9a-fA-F]*|[0-9]+")
_DIGITS = re.compile(r"[0-9]{1,5}")
_SCHEME_PREFIX = re.compile(r"[A-Za-z][A-Za-z0-9+.\-]*:")
_DOT_VARIANTS = {"。": ".", "｡": ".", "．": "."}
_ENTRY_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,63}")


def _hard_name_reason(name: str) -> Optional[Reason]:
    if name in METADATA_NAMES:
        return Reason.DENY_METADATA_NAME
    if any(name.endswith(suffix) or (name == suffix[1:] and name != "localhost") for suffix in INTERNAL_SUFFIXES):
        return Reason.DENY_INTERNAL_NAME
    return None


def _parse_numeric_ipv4(labels: Sequence[str]) -> Any:
    """The WHATWG numeric IPv4 parser: 1 to 4 parts, each decimal, ``0x`` hex or leading-zero octal, the last part
    filling the remaining bytes. Anything it cannot parse is a refusal (never 'a name')."""
    if not 1 <= len(labels) <= 4:
        raise Refused(Reason.DENY_HOST_SYNTAX)
    numbers: List[int] = []
    for part in labels:
        if _NUMERIC_LABEL.fullmatch(part) is None:
            raise Refused(Reason.DENY_HOST_SYNTAX)
        if part[:2] in ("0x", "0X"):
            digits = part[2:]
            numbers.append(int(digits, 16) if digits else 0)
        elif len(part) > 1 and part[0] == "0":
            if any(c in "89" for c in part):
                raise Refused(Reason.DENY_HOST_SYNTAX)
            numbers.append(int(part, 8))
        else:
            numbers.append(int(part, 10))
    *head, last = numbers
    if any(n > 255 for n in head) or last >= 256 ** (5 - len(numbers)):
        raise Refused(Reason.DENY_HOST_SYNTAX)
    value = last
    for i, n in enumerate(head):
        value += n * 256 ** (3 - i)
    return ipaddress.IPv4Address(value)


@dataclass(frozen=True)
class Target:
    """A parsed, normalised request target: only the parts a decision may use."""

    scheme: str
    host: str                       # lowercase name, dotted-decimal IPv4 or full-form IPv6 (no brackets)
    port: int
    kind: str                       # "NAME", "IPV4" or "IPV6"
    address: Any = None             # the ipaddress object for a literal, else None
    odd_form: bool = False          # the spelling differed from the canonical form (encoding, numeric form, dot, case)

    @property
    def origin(self) -> str:
        host = f"[{self.host}]" if self.kind == "IPV6" else self.host
        return f"{self.scheme}://{host}:{self.port}"


def _normalise_host(raw: str) -> Tuple[str, bool]:
    odd = False
    host = raw
    if "%" in host:
        try:
            host = unquote_to_bytes(host).decode("utf-8")
        except UnicodeDecodeError:
            raise Refused(Reason.DENY_HOST_SYNTAX) from None
        if "%" in host:                                  # two levels of encoding: refuse rather than guess
            raise Refused(Reason.DENY_HOST_SYNTAX)
        odd = True
    folded = unicodedata.normalize("NFKC", host)
    for variant, plain in _DOT_VARIANTS.items():
        folded = folded.replace(variant, plain)
    if folded != host:
        odd = True
    host = folded
    if not host.isascii():
        try:
            host = host.encode("idna").decode("ascii")
        except UnicodeError:
            raise Refused(Reason.DENY_HOST_SYNTAX) from None
        odd = True
    if host != host.lower():
        host, odd = host.lower(), True
    if host.endswith("."):
        host, odd = host[:-1], True
    if not host or host.endswith(".") or len(host) > MAX_HOST_CHARS:
        raise Refused(Reason.DENY_HOST_SYNTAX)
    return host, odd


def _parse_host(raw: str, bracketed: bool) -> Tuple[str, str, Any, bool]:
    """-> (host, kind, address, odd_form)."""
    if bracketed:
        text = raw.strip()
        if not text or "%" in text or text != raw or any(c not in "0123456789abcdefABCDEF:." for c in text):
            raise Refused(Reason.DENY_HOST_SYNTAX)
        try:
            addr = ipaddress.IPv6Address(text)
        except ValueError:
            raise Refused(Reason.DENY_HOST_SYNTAX) from None
        return address_text(addr), "IPV6", addr, text != text.lower()
    host, odd = _normalise_host(raw)
    labels = host.split(".")
    if _NUMERIC_LABEL.fullmatch(labels[-1]) is not None:
        addr = _parse_numeric_ipv4(labels)
        return str(addr), "IPV4", addr, odd or str(addr) != host
    if len(labels) > 127 or any(_LABEL.fullmatch(label) is None for label in labels):
        raise Refused(Reason.DENY_HOST_SYNTAX)
    return host, "NAME", None, odd


def _check_url_text(url: Any) -> str:
    if type(url) is not str or not url or len(url) > MAX_URL_CHARS:
        raise Refused(Reason.DENY_MALFORMED_URL)
    for ch in url:
        o = ord(ch)
        if o <= 0x20 or o == 0x7F or ch == "\\":
            raise Refused(Reason.DENY_MALFORMED_URL)
    return url


def parse_target(url: Any) -> Target:
    """Parse and normalise a URL for a decision. Raises ``Refused`` (internal) on anything not admissible."""
    text = _check_url_text(url)
    match = _SCHEME_PREFIX.match(text)
    if match is None:
        raise Refused(Reason.DENY_MALFORMED_URL)
    scheme = match.group(0)[:-1].lower()
    if scheme not in SCHEMES:
        raise Refused(Reason.DENY_SCHEME)
    rest = text[match.end():]
    if not rest.startswith("//"):
        raise Refused(Reason.DENY_MALFORMED_URL)
    rest = rest[2:]
    end = len(rest)
    for i, ch in enumerate(rest):
        if ch in "/?#":
            end = i
            break
    authority = rest[:end]
    if "@" in authority:
        raise Refused(Reason.DENY_USERINFO)
    if authority.startswith("["):
        close = authority.find("]")
        if close == -1:
            raise Refused(Reason.DENY_HOST_SYNTAX)
        host_raw, tail, bracketed = authority[1:close], authority[close + 1:], True
        if tail and not tail.startswith(":"):
            raise Refused(Reason.DENY_HOST_SYNTAX)
        port_text = tail[1:] if tail else ""
    else:
        if authority.count(":") > 1:
            raise Refused(Reason.DENY_HOST_SYNTAX)          # an IPv6 literal must be bracketed
        host_raw, _, port_text = authority.partition(":")
        bracketed = False
    if not host_raw:
        raise Refused(Reason.DENY_HOST_SYNTAX)
    if port_text == "":
        port = DEFAULT_PORTS[scheme]
    else:
        if _DIGITS.fullmatch(port_text) is None:
            raise Refused(Reason.DENY_PORT)
        port = int(port_text)
        if not 1 <= port <= 65535:
            raise Refused(Reason.DENY_PORT)
    host, kind, addr, odd = _parse_host(host_raw, bracketed)
    return Target(scheme=scheme, host=host, port=port, kind=kind, address=addr, odd_form=odd)


_HOST_PORT_FORBIDDEN = frozenset("/?#@\\ \t\r\n\x00")


def host_port_url(host: Any, port: Any, scheme: Any = "https") -> str:
    """The URL a ``(scheme, host, port)`` triple stands for, for a connector that knows a host and a port and no URL.
    A host that holds a character a URL would read as structure (a slash, ``@``, ``?``, ``#``, a backslash, whitespace)
    is refused rather than composed: ``evil.test/x`` must never become the host ``evil.test``."""
    if type(scheme) is not str or scheme.lower() not in SCHEMES:
        raise Refused(Reason.DENY_SCHEME)
    if type(port) is not int or not 1 <= port <= 65535:
        raise Refused(Reason.DENY_PORT)
    if type(host) is not str or not host or any(c in _HOST_PORT_FORBIDDEN or ord(c) < 0x20 for c in host):
        raise Refused(Reason.DENY_HOST_SYNTAX)
    shown = f"[{host}]" if ":" in host and not host.startswith("[") else host
    return f"{scheme.lower()}://{shown}:{port}/"


def resolve_location(base_url: str, location: Any) -> str:
    """The absolute URL a redirect ``Location`` points at, resolved against ``base_url`` (an already decided URL).

    Only the origin of a relative reference matters to a decision, so a relative reference resolves to the base origin.
    Control characters, spaces and backslashes are refused (a browser would fold a backslash into a slash)."""
    try:
        text = _check_url_text(location)
    except Refused:
        raise Refused(Reason.DENY_REDIRECT_LOCATION) from None
    if _SCHEME_PREFIX.match(text):
        return text
    base = parse_target(base_url)
    if text.startswith("//"):
        return base.scheme + ":" + text
    return base.origin + "/"


# ---------------------------------------------------------------------------------------------------------------
# the operator's allow-list
# ---------------------------------------------------------------------------------------------------------------

def _parse_network(text: Any) -> Any:
    if type(text) is not str or not text or text != text.strip():
        raise SpecInvalid("AllowEntry.networks: expected an address or CIDR string")
    try:
        net = ipaddress.ip_network(text, strict=True)
    except ValueError as exc:
        raise SpecInvalid(f"AllowEntry.networks: {text!r} is not a valid address or network") from exc
    minimum = MIN_PRIVATE_PREFIX_V4 if net.version == 4 else MIN_PRIVATE_PREFIX_V6
    if net.prefixlen < minimum:
        raise SpecInvalid(f"AllowEntry.networks: {text!r} is broader than /{minimum}; list a narrower network")
    return net


def _normalise_entry_host(text: Any, private: bool) -> Tuple[str, bool]:
    """-> (host, wildcard). A wildcard is ``*.<named public domain>`` and is never private."""
    if type(text) is not str or not text or text != text.strip():
        raise SpecInvalid("AllowEntry.host: expected a host string")
    if text.startswith("*."):
        if private:
            raise SpecInvalid("AllowEntry.host: a private_ok entry names one exact host, not a wildcard")
        base = text[2:]
        try:
            host, kind, _, odd = _parse_host(base, False)
        except Refused:
            raise SpecInvalid(f"AllowEntry.host: {text!r} is not a valid wildcard") from None
        if kind != "NAME" or odd or host != base or host.count(".") < 1 or _hard_name_reason(host) is not None:
            raise SpecInvalid(f"AllowEntry.host: a wildcard must name a public domain with at least two labels: {text!r}")
        return "*." + host, True
    bracketed = text.startswith("[") and text.endswith("]")
    try:
        host, kind, _, odd = _parse_host(text[1:-1] if bracketed else text, bracketed)
    except Refused:
        raise SpecInvalid(f"AllowEntry.host: {text!r} is not a valid host") from None
    if kind == "IPV4" and host != text:
        raise SpecInvalid(f"AllowEntry.host: write the address {host!r} in dotted decimal")
    if odd and kind == "NAME":
        raise SpecInvalid(f"AllowEntry.host: write the host {host!r} in lowercase ASCII without a trailing dot")
    if kind == "NAME" and _hard_name_reason(host) is not None:
        raise SpecInvalid(f"AllowEntry.host: {host!r} is a metadata or internal-suffix name and can never be admitted")
    return host, False


@dataclass(frozen=True, kw_only=True)
class AllowEntry:
    """One operator-owned allow-list entry.

    * ``schemes`` and ``ports`` are explicit (no default port is assumed).
    * ``host`` is an exact name or address literal, or ``*.<domain>`` (subdomains of a named public domain only).
    * ``private_ok=True`` admits a target that resolves into ``networks`` (the exact addresses or CIDRs the operator
      names; at least ``/16`` for IPv4 and ``/48`` for IPv6). Such an entry names one exact ``host`` or, with no
      ``host``, matches address literals inside ``networks``. Metadata and link-local space is still refused.
    * a public entry (``private_ok=False``) admits only hosts whose every resolved address is a public address.
    """

    schemes: Tuple[str, ...]
    ports: Tuple[int, ...]
    host: Optional[str] = None
    networks: Tuple[str, ...] = ()
    private_ok: bool = False
    label: str = ""

    def __post_init__(self) -> None:
        if type(self.private_ok) is not bool:
            raise SpecInvalid("AllowEntry.private_ok: expected a boolean")
        schemes = tuple(self.schemes) if isinstance(self.schemes, (tuple, list)) else ()
        if not schemes or any(type(s) is not str or s not in SCHEMES for s in schemes) or len(set(schemes)) != len(schemes):
            raise SpecInvalid("AllowEntry.schemes: expected a non-empty subset of http, https, ws, wss")
        ports = tuple(self.ports) if isinstance(self.ports, (tuple, list)) else ()
        if not ports or any(type(p) is not int or not 1 <= p <= 65535 for p in ports) or len(set(ports)) != len(ports):
            raise SpecInvalid("AllowEntry.ports: expected explicit ports between 1 and 65535")
        if type(self.label) is not str or (self.label and _ENTRY_ID.fullmatch(self.label) is None):
            raise SpecInvalid("AllowEntry.label: expected a short identifier")
        networks: Tuple[str, ...] = ()
        host: Optional[str] = None
        if self.private_ok:
            if not self.networks:
                raise SpecInvalid("AllowEntry: a private_ok entry must name the networks the target may resolve to")
            parsed = sorted({_parse_network(n) for n in self.networks}, key=lambda n: (n.version, int(n.network_address),
                                                                                       n.prefixlen))
            networks = tuple(str(n) for n in parsed)
            if self.host is not None:
                host, _ = _normalise_entry_host(self.host, True)
        else:
            if self.networks:
                raise SpecInvalid("AllowEntry: networks belong to private_ok entries only")
            if self.host is None:
                raise SpecInvalid("AllowEntry: a public entry must name a host")
            host, _ = _normalise_entry_host(self.host, False)
        object.__setattr__(self, "schemes", tuple(sorted(schemes)))
        object.__setattr__(self, "ports", tuple(sorted(ports)))
        object.__setattr__(self, "host", host)
        object.__setattr__(self, "networks", networks)

    # -- matching ------------------------------------------------------------------------------------------
    @property
    def wildcard(self) -> bool:
        return self.host is not None and self.host.startswith("*.")

    def _networks(self) -> Tuple[Any, ...]:
        return tuple(ipaddress.ip_network(n) for n in self.networks)

    def matches(self, target: Target) -> bool:
        """Scheme, port and host (the address check is separate and needs the resolved addresses)."""
        if target.scheme not in self.schemes and _SAME_ORIGIN_ENTRY.get(target.scheme) not in self.schemes:
            return False
        if target.port not in self.ports:
            return False
        if self.host is None:                                   # private entry with networks only: address literals
            return target.kind != "NAME" and any(target.address in n for n in self._networks()
                                                 if n.version == target.address.version)
        if self.host.startswith("*."):
            return target.kind == "NAME" and target.host.endswith(self.host[1:])   # subdomains, never the apex
        return target.host == self.host

    def to_json(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"ports": list(self.ports), "private_ok": self.private_ok, "schemes": list(self.schemes)}
        if self.host is not None:
            out["host"] = self.host
        if self.networks:
            out["networks"] = list(self.networks)
        if self.label:
            out["label"] = self.label
        return out

    @staticmethod
    def parse(text: str, *, label: str = "") -> Tuple["AllowEntry", ...]:
        """Public entries from operator text: ``scheme://host[:port]`` (exact; the scheme's default port if none),
        ``host:port`` (``http`` and ``https`` on that port) or ``host`` (``http`` on 80 and ``https`` on 443). ``host``
        may be ``*.domain``. Operator configuration only, never page content."""
        if type(text) is not str or not text or text != text.strip():
            raise SpecInvalid("AllowEntry.parse: expected a non-empty string")
        scheme: Optional[str] = None
        rest = text
        if "://" in text:
            scheme, _, rest = text.partition("://")
            scheme = scheme.lower()
            if scheme not in SCHEMES:
                raise SpecInvalid(f"AllowEntry.parse: unsupported scheme in {text!r}")
        if not rest or any(c in rest for c in "/?#@\\ \t\r\n"):
            raise SpecInvalid("AllowEntry.parse: an entry is an origin: no path, query, fragment or userinfo")
        if rest.startswith("["):
            close = rest.find("]")
            host, tail = (rest[:close + 1], rest[close + 1:]) if close != -1 else (rest, "")
            port_text = tail[1:] if tail.startswith(":") else (None if not tail else "!")
        elif ":" in rest:
            host, _, port_text = rest.partition(":")
        else:
            host, port_text = rest, None
        if port_text is not None and _DIGITS.fullmatch(port_text) is None:
            raise SpecInvalid(f"AllowEntry.parse: bad port in {text!r}")
        port = int(port_text) if port_text is not None else None
        if scheme is not None:
            return (AllowEntry(schemes=(scheme,), ports=(DEFAULT_PORTS[scheme] if port is None else port,), host=host,
                               label=label),)
        if port is not None:
            return (AllowEntry(schemes=("http", "https"), ports=(port,), host=host, label=label),)
        return (AllowEntry(schemes=("http",), ports=(80,), host=host, label=label),
                AllowEntry(schemes=("https",), ports=(443,), host=host, label=label))


@dataclass(frozen=True, kw_only=True)
class EgressPolicy:
    """The operator's egress configuration: the allow-list and the redirect cap. Default deny: no entry, no egress."""

    entries: Tuple[AllowEntry, ...] = ()
    max_redirects: int = DEFAULT_MAX_REDIRECTS
    max_addresses: int = DEFAULT_MAX_ADDRESSES

    def __post_init__(self) -> None:
        entries = tuple(self.entries)
        if any(type(e) is not AllowEntry for e in entries):
            raise SpecInvalid("EgressPolicy.entries: expected AllowEntry values")
        for name, low, high in (("max_redirects", 0, MAX_REDIRECTS_LIMIT), ("max_addresses", 1, 64)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise SpecInvalid(f"EgressPolicy.{name}: expected an integer between {low} and {high}")
        ordered = sorted(entries, key=lambda e: canon.canonical_text(e.to_json()))
        object.__setattr__(self, "entries", tuple(ordered))

    def to_json(self) -> Dict[str, Any]:
        return {"entries": [e.to_json() for e in self.entries], "form": POLICY_FORM, "max_addresses": self.max_addresses,
                "max_redirects": self.max_redirects, "rules_version": EGRESS_RULES_VERSION}

    @property
    def policy_id(self) -> str:
        """``rcv1:<64 hex>`` of the policy and the rule version: the value for ``RunConfig.egress_policy_id``."""
        return canon.to_contract_ref(canon.canon_hash(self.to_json()))

    @staticmethod
    def from_config(config: Mapping[str, Any]) -> "EgressPolicy":
        """Build a policy from operator configuration (JSON-shaped): ``allow`` (entry strings for
        :meth:`AllowEntry.parse`), ``wildcards`` (``{host, schemes, ports}``), ``private`` (``{host?, networks, schemes,
        ports, label?}``), ``max_redirects``. Unknown keys are refused."""
        if not isinstance(config, Mapping):
            raise SpecInvalid("EgressPolicy.from_config: expected a mapping")
        unknown = set(config) - {"allow", "wildcards", "private", "max_redirects", "max_addresses"}
        if unknown:
            raise SpecInvalid(f"EgressPolicy.from_config: unknown keys {sorted(map(str, unknown))}")
        entries: List[AllowEntry] = []
        try:
            for text in config.get("allow", ()):
                entries.extend(AllowEntry.parse(text))
            for item in config.get("wildcards", ()):
                entries.append(AllowEntry(host=item["host"], schemes=tuple(item["schemes"]), ports=tuple(item["ports"]),
                                          label=item.get("label", "")))
            for item in config.get("private", ()):
                entries.append(AllowEntry(host=item.get("host"), networks=tuple(item["networks"]),
                                          schemes=tuple(item["schemes"]), ports=tuple(item["ports"]), private_ok=True,
                                          label=item.get("label", "")))
        except (KeyError, TypeError, AttributeError) as exc:
            raise SpecInvalid(f"EgressPolicy.from_config: malformed entry ({type(exc).__name__})") from exc
        kwargs = {k: config[k] for k in ("max_redirects", "max_addresses") if k in config}
        return EgressPolicy(entries=tuple(entries), **kwargs)


# ---------------------------------------------------------------------------------------------------------------
# resolvers (injected; none here touches the network)
# ---------------------------------------------------------------------------------------------------------------

@runtime_checkable
class Resolver(Protocol):
    """Name resolution, injected. ``resolve`` returns address literals (text) for ``host``; it may raise."""

    def resolve(self, host: str) -> Sequence[str]: ...


class NoResolver:
    """The default: refuses every name. A guard without a real resolver can only decide for address literals."""

    def resolve(self, host: str) -> Sequence[str]:
        raise LookupError("no resolver is configured")


class StaticResolver:
    """A fixed table, for tests and for a gateway that resolves elsewhere. Unknown names raise (not 'no addresses')."""

    def __init__(self, table: Mapping[str, Sequence[str]]) -> None:
        self._table = {str(k).lower(): tuple(v) for k, v in table.items()}
        self.calls: List[str] = []

    def resolve(self, host: str) -> Sequence[str]:
        self.calls.append(host)
        try:
            return self._table[host]
        except KeyError:
            raise LookupError("unknown host") from None


# ---------------------------------------------------------------------------------------------------------------
# the decision record
# ---------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True)
class EgressDecision:
    """One decision, allow or deny, as plain data with a ``canon-v1`` hash. No URL path, query or fragment is kept."""

    verdict: EgressVerdict
    reasons: Tuple[str, ...]
    purpose: str
    hop: int
    policy_id: str
    scheme: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    host_kind: Optional[str] = None
    addresses: Tuple[str, ...] = ()
    address_classes: Tuple[str, ...] = ()
    pinned_address: Optional[str] = None
    mode: str = "DENIED"                 # "PUBLIC", "PRIVATE_OPERATOR" or "DENIED"
    entry: Optional[str] = None          # the matched entry's label or its canonical hash
    resolved: bool = False               # False: no addresses were checked (a static listing decision)
    odd_form: bool = False

    @property
    def allowed(self) -> bool:
        return self.verdict is EgressVerdict.ALLOW

    def to_json(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "address_classes": list(self.address_classes), "addresses": list(self.addresses), "form": DECISION_FORM,
            "hop": self.hop, "mode": self.mode, "odd_form": self.odd_form, "policy_id": self.policy_id,
            "purpose": self.purpose, "reasons": list(self.reasons), "resolved": self.resolved,
            "rules_version": EGRESS_RULES_VERSION, "verdict": self.verdict.value}
        for key in ("scheme", "host", "port", "host_kind", "pinned_address", "entry"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        return out

    @property
    def record_hash(self) -> str:
        """``canon-v1:<64 hex>`` of the decision."""
        return canon.canon_hash(self.to_json())

    def ref(self) -> str:
        """The same digest in the contracts spelling, ``rcv1:<64 hex>``."""
        return canon.to_contract_ref(self.record_hash)

    def connect_target(self) -> Tuple[str, int, str]:
        """``(address, port, server_name)`` a connector must use: the pinned address, never a fresh resolution.
        Raises ``PermissionError`` for a denied decision."""
        if not self.allowed or self.pinned_address is None or self.port is None or self.host is None:
            raise PermissionError("egress denied: there is nothing to connect to")
        return self.pinned_address, self.port, self.host

    def permits_peer(self, peer_address: Any) -> bool:
        """True only if the connection actually made went to the pinned address (DNS-rebinding check)."""
        if not self.allowed or self.pinned_address is None:
            return False
        try:
            peer = ipaddress.ip_address(peer_address) if not hasattr(peer_address, "version") else peer_address
        except ValueError:
            return False
        return address_text(peer) == self.pinned_address


def _entry_name(entry: AllowEntry, index: int) -> str:
    return entry.label or f"entry-{index}"


def _deny(policy_id: str, reasons: Iterable[Reason], purpose: str, hop: int, target: Optional[Target] = None, *,
          addresses: Sequence[Any] = (), classes: Iterable[AddressClass] = (), resolved: bool = False) -> EgressDecision:
    return EgressDecision(
        verdict=EgressVerdict.DENY, reasons=tuple(sorted({r.value for r in reasons})), purpose=purpose, hop=hop,
        policy_id=policy_id, scheme=target.scheme if target else None, host=target.host if target else None,
        port=target.port if target else None, host_kind=target.kind if target else None,
        addresses=tuple(address_text(a) for a in addresses), address_classes=tuple(sorted({c.value for c in classes})),
        resolved=resolved, odd_form=target.odd_form if target else False)


def _parse_answer(item: Any) -> Any:
    if type(item) is not str or not item or item != item.strip() or "%" in item:
        raise Refused(Reason.DENY_RESOLVER_BAD_ANSWER)
    try:
        return ipaddress.ip_address(item)
    except ValueError:
        raise Refused(Reason.DENY_RESOLVER_BAD_ANSWER) from None


def _sorted_unique(addresses: Iterable[Any]) -> List[Any]:
    unique = {address_text(a): a for a in addresses}
    return sorted(unique.values(), key=lambda a: (a.version, int(a)))


_CLASS_REASON = {AddressClass.MAPPED: Reason.DENY_ADDRESS_MAPPED, AddressClass.TRANSITION: Reason.DENY_ADDRESS_TRANSITION}


def decide(policy: EgressPolicy, target: Union[str, Target], addresses: Optional[Sequence[Any]] = None, *,
           purpose: str = "navigation", hop: int = 0, previous_scheme: Optional[str] = None) -> EgressDecision:
    """The decision for ONE request, pure. ``addresses`` are the addresses the (injected) resolver returned for a name
    target; pass ``None`` for no resolution (then a listed name host is admitted on the listing alone and the record
    says ``resolved: false``). For an address literal the literal is the address, whatever else is passed.

    ``previous_scheme`` is the scheme of the hop that redirected here (an ``https`` to ``http`` downgrade is denied).

    Order: parse and normalise; scheme, userinfo, port, host syntax; metadata and internal names; metadata addresses;
    a scheme downgrade; the allow-list (default deny); then the address rules of the entry that matched."""
    pid = policy.policy_id
    parsed: Optional[Target] = None
    try:
        parsed = parse_target(target) if not isinstance(target, Target) else target
        name_reason = _hard_name_reason(parsed.host) if parsed.kind == "NAME" else None
        if name_reason is not None:
            return _deny(pid, [name_reason], purpose, hop, parsed)
        # --- addresses ---------------------------------------------------------------------------------
        resolved = True
        if parsed.kind != "NAME":
            found = [parsed.address]
        elif addresses is None:
            found, resolved = [], False
        else:
            if not isinstance(addresses, (list, tuple)) or not addresses:
                return _deny(pid, [Reason.DENY_NO_ADDRESSES], purpose, hop, parsed, resolved=True)
            if len(addresses) > policy.max_addresses:
                return _deny(pid, [Reason.DENY_TOO_MANY_ADDRESSES], purpose, hop, parsed, resolved=True)
            found = [a if hasattr(a, "version") else _parse_answer(a) for a in addresses]
        ordered = _sorted_unique(found)
        classes = {a: classify_address(a) for a in ordered}
        if any(c is AddressClass.METADATA or c is AddressClass.LINK_LOCAL for c in classes.values()):
            # refused always: this runs before the allow-list is consulted, so no entry can admit it
            return _deny(pid, [Reason.DENY_METADATA_ADDRESS], purpose, hop, parsed, addresses=ordered,
                         classes=classes.values(), resolved=resolved)
        if previous_scheme is not None and _downgrade(previous_scheme, parsed.scheme):
            return _deny(pid, [Reason.DENY_SCHEME_DOWNGRADE], purpose, hop, parsed, addresses=ordered,
                         classes=classes.values(), resolved=resolved)
        # --- the allow-list (default deny) -------------------------------------------------------------
        matching = [(i, e) for i, e in enumerate(policy.entries) if e.matches(parsed)]
        if not matching:
            return _deny(pid, [Reason.DENY_NOT_LISTED], purpose, hop, parsed, addresses=ordered,
                         classes=classes.values(), resolved=resolved)
        refusals: List[Reason] = []
        for index, entry in matching:
            problem = _address_problem(entry, ordered, classes, resolved)
            if not problem:
                reasons = [Reason.ALLOWED_PRIVATE_OPERATOR if entry.private_ok else Reason.ALLOWED_PUBLIC]
                if not resolved:
                    reasons.append(Reason.STATIC_ONLY)
                return EgressDecision(
                    verdict=EgressVerdict.ALLOW, reasons=tuple(sorted(r.value for r in reasons)), purpose=purpose,
                    hop=hop, policy_id=pid, scheme=parsed.scheme, host=parsed.host, port=parsed.port,
                    host_kind=parsed.kind, addresses=tuple(address_text(a) for a in ordered),
                    address_classes=tuple(sorted({c.value for c in classes.values()})),
                    pinned_address=address_text(ordered[0]) if ordered else None,
                    mode="PRIVATE_OPERATOR" if entry.private_ok else "PUBLIC", entry=_entry_name(entry, index),
                    resolved=resolved, odd_form=parsed.odd_form)
            refusals.extend(problem)
        return _deny(pid, refusals, purpose, hop, parsed, addresses=ordered, classes=classes.values(), resolved=resolved)
    except Refused as refusal:
        return _deny(pid, [refusal.reason], purpose, hop, parsed)


def _address_problem(entry: AllowEntry, ordered: Sequence[Any], classes: Mapping[Any, AddressClass],
                     resolved: bool) -> List[Reason]:
    """Why ``entry`` cannot admit these addresses (empty: it can). Every address must pass, not just one."""
    if not resolved:
        return []
    problems: List[Reason] = []
    if entry.private_ok:
        nets = entry._networks()
        for addr in ordered:
            cls = classes[addr]
            if cls in _CLASS_REASON:
                problems.append(_CLASS_REASON[cls])
            elif not any(addr in n for n in nets if n.version == addr.version):
                problems.append(Reason.DENY_ADDRESS_OUTSIDE_ENTRY)
        return problems
    for addr in ordered:
        cls = classes[addr]
        if cls is not AddressClass.PUBLIC:
            problems.append(_CLASS_REASON.get(cls, Reason.DENY_ADDRESS_NOT_PUBLIC))
    return problems


def decide_host_port(policy: EgressPolicy, host: Any, port: Any, addresses: Optional[Sequence[Any]] = None, *,
                     scheme: Any = "https", purpose: str = "connection", hop: int = 0) -> EgressDecision:
    """:func:`decide` for a ``(host, port)`` pair (a CONNECT target, a socket destination) instead of a URL."""
    try:
        url = host_port_url(host, port, scheme)
    except Refused as refusal:
        return _deny(policy.policy_id, [refusal.reason], purpose, hop)
    return decide(policy, url, addresses, purpose=purpose, hop=hop)


# ---------------------------------------------------------------------------------------------------------------
# redirect chains
# ---------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ChainDecision:
    """The decisions of a request and its redirects, evaluated hop by hop and stopped at the first denial."""

    decisions: Tuple[EgressDecision, ...]

    @property
    def allowed(self) -> bool:
        return bool(self.decisions) and all(d.allowed for d in self.decisions)

    @property
    def final(self) -> EgressDecision:
        return self.decisions[-1]

    @property
    def chain_hash(self) -> str:
        return canon.canon_hash({"decisions": [d.record_hash for d in self.decisions], "form": "qa-egress-chain/1"})


def decide_chain(policy: EgressPolicy, hops: Sequence[Tuple[str, Optional[Sequence[Any]]]], *,
                 purpose: str = "navigation") -> ChainDecision:
    """Decide a request and its redirect chain. ``hops[0]`` is the request ``(url, resolved addresses)``; each next
    item is a redirect ``(Location, resolved addresses of the target)``. A relative ``Location`` is resolved against
    the previous hop. **Every hop is decided again from scratch**; the first denial ends the chain; more than
    ``policy.max_redirects`` redirects is a denial; an ``https`` to ``http`` downgrade is a denial."""
    if not hops:
        raise SpecInvalid("decide_chain: no hops")
    decisions: List[EgressDecision] = []
    previous_url: Optional[str] = None
    previous_scheme: Optional[str] = None
    for index, (url, addresses) in enumerate(hops):
        if index > policy.max_redirects:
            decisions.append(_deny(policy.policy_id, [Reason.DENY_REDIRECT_LIMIT], "redirect", index))
            break
        try:
            absolute = url if previous_url is None else resolve_location(previous_url, url)
        except Refused as refusal:
            decisions.append(_deny(policy.policy_id, [refusal.reason], "redirect" if index else purpose, index))
            break
        decision = decide(policy, absolute, addresses, purpose=purpose if index == 0 else "redirect", hop=index,
                          previous_scheme=previous_scheme)
        decisions.append(decision)
        if not decision.allowed:
            break
        previous_url, previous_scheme = absolute, decision.scheme
    return ChainDecision(decisions=tuple(decisions))


# ---------------------------------------------------------------------------------------------------------------
# the guard: resolver + policy + the decision log
# ---------------------------------------------------------------------------------------------------------------

class EgressGuard:
    """A policy plus an injected resolver, with a log of every decision made through it.

    One guard belongs to one run: the runner drains its log into the evidence stream (``drain``). The guard calls the
    resolver once per decision and never keeps a pin between decisions: every new connection is a new decision."""

    def __init__(self, policy: EgressPolicy, resolver: Optional[Resolver] = None) -> None:
        if type(policy) is not EgressPolicy:
            raise SpecInvalid("EgressGuard: policy must be an EgressPolicy")
        if resolver is not None and not callable(getattr(resolver, "resolve", None)):
            raise SpecInvalid("EgressGuard: resolver must provide resolve(host)")
        self._policy = policy
        self._resolver: Resolver = resolver if resolver is not None else NoResolver()
        self._pending: List[EgressDecision] = []
        self._allowed = 0
        self._denied = 0
        self._dropped = 0
        self._denied_hosts: Dict[str, int] = {}

    @property
    def policy(self) -> EgressPolicy:
        return self._policy

    @property
    def policy_id(self) -> str:
        return self._policy.policy_id

    # -- bookkeeping ---------------------------------------------------------------------------------------
    def _record(self, decision: EgressDecision) -> EgressDecision:
        if decision.allowed:
            self._allowed += 1
        else:
            self._denied += 1
            key = decision.host or "(unparsed)"
            self._denied_hosts[key] = self._denied_hosts.get(key, 0) + 1
        if len(self._pending) >= MAX_PENDING_DECISIONS:
            self._dropped += 1
        else:
            self._pending.append(decision)
        return decision

    def drain(self) -> Tuple[EgressDecision, ...]:
        """The decisions made since the last drain, in order; the log is emptied."""
        out, self._pending = tuple(self._pending), []
        return out

    @property
    def counts(self) -> Dict[str, int]:
        return {"allowed": self._allowed, "denied": self._denied, "dropped": self._dropped}

    def denied_hosts(self) -> Tuple[Tuple[str, int], ...]:
        """Denied hosts with their counts, sorted by host (hosts only: never a path, query or fragment)."""
        return tuple(sorted(self._denied_hosts.items()))

    # -- decisions -----------------------------------------------------------------------------------------
    def _resolve(self, target: Target) -> Union[List[str], Reason]:
        try:
            answer = self._resolver.resolve(target.host)
        except Exception:                                    # the message is not kept: it could echo anything
            return Reason.DENY_RESOLVER_ERROR
        if isinstance(answer, (str, bytes)) or not isinstance(answer, (list, tuple)):
            return Reason.DENY_RESOLVER_BAD_ANSWER
        return list(answer)

    def check(self, url: Any, *, purpose: str = "subresource", hop: int = 0,
              previous_scheme: Optional[str] = None) -> EgressDecision:
        """Decide one request: parse, resolve once (names only), decide on all the addresses, pin, log."""
        pid = self._policy.policy_id
        try:
            target = parse_target(url)
        except Refused as refusal:
            return self._record(_deny(pid, [refusal.reason], purpose, hop))
        if target.kind == "NAME" and _hard_name_reason(target.host) is None and self._listed(target):
            answer = self._resolve(target)
            if isinstance(answer, Reason):
                return self._record(_deny(pid, [answer], purpose, hop, target, resolved=True))
            return self._record(decide(self._policy, target, answer, purpose=purpose, hop=hop,
                                       previous_scheme=previous_scheme))
        return self._record(decide(self._policy, target, None, purpose=purpose, hop=hop,
                                   previous_scheme=previous_scheme))

    def _listed(self, target: Target) -> bool:
        """Resolution is only worth doing for a name some entry matches; an unlisted name is denied without a lookup
        (no DNS query leaves for a host the operator never listed)."""
        return any(e.matches(target) for e in self._policy.entries)

    def check_host_port(self, host: Any, port: Any, *, scheme: Any = "https", purpose: str = "connection") -> EgressDecision:
        """:meth:`check` for a ``(host, port)`` pair."""
        try:
            url = host_port_url(host, port, scheme)
        except Refused as refusal:
            return self._record(_deny(self._policy.policy_id, [refusal.reason], purpose, 0))
        return self.check(url, purpose=purpose)

    def check_static(self, url: Any, *, purpose: str = "observed-url") -> EgressDecision:
        """The listing-only decision for a URL already reached (no resolution; ``resolved: false`` when admitted)."""
        pid = self._policy.policy_id
        try:
            target = parse_target(url)
        except Refused as refusal:
            return self._record(_deny(pid, [refusal.reason], purpose, 0))
        return self._record(decide(self._policy, target, None, purpose=purpose))

    def check_redirect(self, previous_url: str, location: Any, *, hop: int,
                       previous: Optional[EgressDecision] = None) -> EgressDecision:
        """Decide a redirect hop from scratch. ``previous_url`` is the URL that answered with the redirect, ``hop`` is
        the 1-based hop number and ``previous`` (its decision) gives the scheme for the downgrade rule."""
        pid = self._policy.policy_id
        if hop > self._policy.max_redirects:
            return self._record(_deny(pid, [Reason.DENY_REDIRECT_LIMIT], "redirect", hop))
        try:
            absolute = resolve_location(previous_url, location)
        except Refused as refusal:
            return self._record(_deny(pid, [refusal.reason], "redirect", hop))
        return self.check(absolute, purpose="redirect", hop=hop,
                          previous_scheme=previous.scheme if previous is not None else None)

    def verify_peer(self, decision: EgressDecision, peer_address: Any, *, purpose: str = "connection") -> EgressDecision:
        """Check the address a connection really reached against the pinned one. Returns an ``ALLOW`` decision equal
        to ``decision`` (logged) or a ``DENY_PEER_MISMATCH`` decision (logged); the connector must close on a denial."""
        if decision.allowed and decision.permits_peer(peer_address):
            return decision
        pid = self._policy.policy_id
        return self._record(EgressDecision(
            verdict=EgressVerdict.DENY, reasons=(Reason.DENY_PEER_MISMATCH.value,), purpose=purpose, hop=decision.hop,
            policy_id=pid, scheme=decision.scheme, host=decision.host, port=decision.port, host_kind=decision.host_kind,
            addresses=decision.addresses, address_classes=decision.address_classes, resolved=True))


__all__ = ["EGRESS_RULES_VERSION", "DECISION_FORM", "POLICY_FORM", "SCHEMES", "DEFAULT_PORTS", "ABOUT_BLANK", "Reason",
           "EgressVerdict", "AddressClass", "Target", "AllowEntry", "EgressPolicy", "Resolver", "NoResolver",
           "StaticResolver", "EgressDecision", "ChainDecision", "EgressGuard", "address_text", "classify_address",
           "is_metadata_address", "parse_target", "resolve_location", "decide", "decide_chain", "decide_host_port",
           "host_port_url", "Refused", "METADATA_NAMES",
           "INTERNAL_SUFFIXES"]

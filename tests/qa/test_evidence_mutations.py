# SPDX-License-Identifier: AGPL-3.0-or-later
"""Mutation tests. Each guard of the egress policy, the redactor, the blob store, the vault and the evidence checks is
removed from a copy of its source (an exact-text replacement that must match once); a probe that holds on the real code
must fail on the mutant: remove the metadata refusal, the resolve-then-pin, the redirect re-check, the userinfo refusal,
the fail-closed redaction, a size bound, a rule ... and the suite goes red. Probes check behaviour, not text."""
import importlib
import json

import pytest

from agent_harness.qa import blobs as B_REAL, canon, records as R, receipts as RC
from agent_harness.qa.errors import RedactionFailed, ResourceLimitExceeded, SpecInvalid
from agent_harness.qa.vault import EvidenceGuard

from . import _fx_evidence as H
from ._fx_verify import Mutants
from ._fx_runner import APP, default_policy, make_env, make_spec, nav_step, operations, save_step
from .test_redaction import ENT, RULE_CASES


@pytest.fixture
def mutants():
    m = Mutants()
    yield m
    m.close()


def real(filename):
    return importlib.import_module("agent_harness.qa." + filename[:-3])


def holds(probe, module):
    try:
        return bool(probe(module))
    except Exception:
        return False


PUB = "93.184.216.34"
TABLE = {"app.example.test": [PUB], "meta.example.test": ["169.254.169.254"], "priv.example.test": ["10.0.0.5"],
         "mixed.example.test": [PUB, "::1"], "mapped.example.test": ["::ffff:93.184.216.34"], "compose-app": ["10.20.0.6"],
         "unlisted.example.test": [PUB], "a.example.test": [PUB], "example.test": [PUB], "xexample.test": [PUB],
         "xn--bcher-kva.example.test": [PUB]}


def entry(E, host, schemes=("http", "https"), ports=(80, 443), **kw):
    return E.AllowEntry(schemes=schemes, ports=ports, host=host, **kw)


def guard(E, *entries, table=None, **kw):
    return E.EgressGuard(E.EgressPolicy(entries=entries, **kw), E.StaticResolver(TABLE if table is None else table))


def hosts(E, *names):
    return [entry(E, n) for n in names]


# =====================================================================================================================
# probes: egress
# =====================================================================================================================

def p_metadata_refused_even_if_listed(E):
    private = E.AllowEntry(schemes=("http",), ports=(80,), host="169.254.169.254", networks=("169.254.169.254",), private_ok=True)
    g = guard(E, private, entry(E, "meta.example.test"))
    a, b = g.check("http://169.254.169.254/"), g.check("http://meta.example.test/")
    return not a.allowed and not b.allowed and "DENY_METADATA_ADDRESS" in a.reasons and "DENY_METADATA_ADDRESS" in b.reasons


def p_embedded_metadata(E):
    g = guard(E)
    return all("DENY_METADATA_ADDRESS" in g.check(u).reasons for u in
               ("http://[::ffff:169.254.169.254]/", "http://[64:ff9b::a9fe:a9fe]/", "http://[2002:a9fe:a9fe::1]/"))


def p_metadata_names(E):
    g = guard(E)
    return "DENY_METADATA_NAME" in g.check("http://metadata.google.internal/").reasons and \
        "DENY_METADATA_NAME" in g.check("http://METADATA./").reasons


def p_internal_names(E):
    g = guard(E)
    return all("DENY_INTERNAL_NAME" in g.check(u).reasons for u in ("http://db.internal/", "http://printer.local/", "http://a.localhost/"))


def p_pin(E):
    g = guard(E, entry(E, "app.example.test"))
    d = g.check("http://app.example.test/")
    return d.allowed and d.pinned_address == PUB and d.connect_target() == (PUB, 80, "app.example.test")


def p_peer(E):
    d = guard(E, entry(E, "app.example.test")).check("http://app.example.test/")
    return d.permits_peer(PUB) is True and d.permits_peer("10.0.0.5") is False and d.permits_peer("169.254.169.254") is False


def p_verify_peer(E):
    g = guard(E, entry(E, "app.example.test"))
    d = g.check("http://app.example.test/")
    return g.verify_peer(d, PUB).allowed and not g.verify_peer(d, "10.0.0.5").allowed


def p_redirect_chain(E):
    policy = E.EgressPolicy(entries=tuple(hosts(E, "app.example.test")))
    d = E.decide_chain(policy, [("http://app.example.test/", [PUB]), ("http://169.254.169.254/", None)])
    return not d.allowed and len(d.decisions) == 2 and "DENY_METADATA_ADDRESS" in d.final.reasons


def p_redirect_chain_unlisted(E):
    policy = E.EgressPolicy(entries=tuple(hosts(E, "app.example.test")))
    d = E.decide_chain(policy, [("http://app.example.test/", [PUB]), ("http://unlisted.example.test/", [PUB])])
    return not d.allowed and "DENY_NOT_LISTED" in d.final.reasons


def p_redirect_guard(E):
    g = guard(E, *hosts(E, "app.example.test"))
    first = g.check("http://app.example.test/")
    d = g.check_redirect("http://app.example.test/", "http://169.254.169.254/latest/", hop=1, previous=first)
    ok = g.check_redirect("http://app.example.test/", "/next", hop=1, previous=first)
    return not d.allowed and "DENY_METADATA_ADDRESS" in d.reasons and ok.allowed


def p_hop_cap_chain(E):
    policy = E.EgressPolicy(entries=tuple(hosts(E, "app.example.test")), max_redirects=2)
    hop = ("http://app.example.test/", [PUB])
    return E.decide_chain(policy, [hop] * 3).allowed and "DENY_REDIRECT_LIMIT" in E.decide_chain(policy, [hop] * 4).final.reasons


def p_hop_cap_guard(E):
    g = guard(E, *hosts(E, "app.example.test"), max_redirects=1)
    return g.check_redirect("http://app.example.test/", "/x", hop=1).allowed and \
        g.check_redirect("http://app.example.test/", "/x", hop=2).reasons == ("DENY_REDIRECT_LIMIT",)


def p_downgrade(E):
    policy = E.EgressPolicy(entries=tuple(hosts(E, "app.example.test")))
    d = E.decide_chain(policy, [("https://app.example.test/", [PUB]), ("http://app.example.test/", [PUB])])
    return not d.allowed and "DENY_SCHEME_DOWNGRADE" in d.final.reasons


def p_downgrade_guard(E):
    g = guard(E, *hosts(E, "app.example.test"))
    first = g.check("https://app.example.test/")
    return "DENY_SCHEME_DOWNGRADE" in g.check_redirect("https://app.example.test/", "http://app.example.test/", hop=1, previous=first).reasons


def p_userinfo(E):
    g = guard(E, *hosts(E, "app.example.test"))
    return all("DENY_USERINFO" in g.check(u).reasons for u in ("http://user" + H.AT + "app.example.test/", "http://u:" + H.gen("m-pw", 8) + H.AT + "app.example.test/",
                                                               "http://app.example.test:80" + H.AT + "evil.test/"))


def p_default_deny(E):
    g = guard(E, *hosts(E, "app.example.test"))
    d = g.check("http://unlisted.example.test/")
    return not d.allowed and d.reasons == ("DENY_NOT_LISTED",) and not E.EgressGuard(E.EgressPolicy()).check(f"http://{PUB}/").allowed


def p_private_networks(E):
    private = E.AllowEntry(schemes=("http",), ports=(8080,), host="compose-app", networks=("10.20.0.5",), private_ok=True)
    g = guard(E, private)
    return "DENY_ADDRESS_OUTSIDE_ENTRY" in g.check("http://compose-app:8080/").reasons


def p_private_ok_when_inside(E):
    private = E.AllowEntry(schemes=("http",), ports=(8080,), host="compose-app", networks=("10.20.0.0/24",), private_ok=True)
    return guard(E, private).check("http://compose-app:8080/").allowed


def p_public_entry_private_address(E):
    g = guard(E, *hosts(E, "priv.example.test"))
    return "DENY_ADDRESS_NOT_PUBLIC" in g.check("http://priv.example.test/").reasons


def p_split_answer(E):
    g = guard(E, *hosts(E, "mixed.example.test"))
    return not g.check("http://mixed.example.test/").allowed


def p_mapped(E):
    g = guard(E, *hosts(E, "mapped.example.test"))
    return "DENY_ADDRESS_MAPPED" in g.check("http://mapped.example.test/").reasons


def p_wildcard(E):
    g = guard(E, E.AllowEntry(schemes=("http",), ports=(80,), host="*.example.test"))
    return g.check("http://a.example.test/").allowed and not g.check("http://example.test/").allowed and \
        not g.check("http://xexample.test/").allowed


def p_numeric_octal(E):
    g = guard(E, entry(E, "127.0.0.1"))
    return "DENY_ADDRESS_NOT_PUBLIC" in g.check("http://0177.0.0.1/").reasons and "DENY_ADDRESS_NOT_PUBLIC" in g.check("http://017700000001/").reasons


def p_numeric_hex(E):
    g = guard(E, entry(E, "127.0.0.1"))
    return "DENY_ADDRESS_NOT_PUBLIC" in g.check("http://0x7f.1/").reasons and "DENY_ADDRESS_NOT_PUBLIC" in g.check("http://0x7f000001/").reasons


def p_numeric_decimal_short(E):
    g = guard(E, entry(E, "127.0.0.1"))
    return "DENY_ADDRESS_NOT_PUBLIC" in g.check("http://2130706433/").reasons and "DENY_ADDRESS_NOT_PUBLIC" in g.check("http://127.1/").reasons


def p_numeric_range(E):
    g = guard(E)
    return all("DENY_HOST_SYNTAX" in g.check(u).reasons for u in ("http://256.1.1.1/", "http://1.2.3.4.5/", "http://08.0.0.1/"))


def p_scheme(E):
    g = guard(E, *hosts(E, "app.example.test"))
    return all(g.check(u).reasons == ("DENY_SCHEME",) for u in ("file:///etc/passwd", "ftp://app.example.test/", "gopher://app.example.test/"))


def p_port(E):
    g = guard(E, *hosts(E, "app.example.test"))
    return all("DENY_PORT" in g.check(f"http://app.example.test:{p}/").reasons for p in ("0", "65536", "99999"))


def p_no_dns_for_unlisted(E):
    g = guard(E, *hosts(E, "app.example.test"))
    g.check("http://unlisted.example.test/")
    return g.check("http://app.example.test/").allowed and g._resolver.calls == ["app.example.test"]


def p_idna(E):
    return guard(E, entry(E, "xn--bcher-kva.example.test")).check("http://bücher.example.test/").allowed


def p_percent(E):
    return guard(E, *hosts(E, "app.example.test")).check("http://%61pp.example.test/").allowed


def p_nfkc(E):
    g = guard(E, entry(E, "127.0.0.1"))
    return "DENY_ADDRESS_NOT_PUBLIC" in g.check("http://１２７.０.０.１/").reasons


def p_trailing_dot(E):
    return guard(E, *hosts(E, "app.example.test")).check("http://app.example.test./").allowed


def p_private_entry_breadth(E):
    for broad in ("10.0.0.0/8", "0.0.0.0/0"):
        try:
            E.AllowEntry(schemes=("http",), ports=(80,), host="x", networks=(broad,), private_ok=True)
        except Exception:
            continue
        return False
    return True


def p_no_resolver_for_literals(E):
    g = E.EgressGuard(E.EgressPolicy(entries=(entry(E, PUB),)))
    return g.check(f"http://{PUB}/").allowed and "DENY_RESOLVER_ERROR" in E.EgressGuard(E.EgressPolicy(entries=tuple(hosts(E, "app.example.test")))).check("http://app.example.test/").reasons


def p_resolver_bad_answer(E):
    g = guard(E, entry(E, "bad.example.test"), table={"bad.example.test": ["not-an-ip"]})
    return "DENY_RESOLVER_BAD_ANSWER" in g.check("http://bad.example.test/").reasons


def p_pending_bound(E):
    g = guard(E)
    for _ in range(E.MAX_PENDING_DECISIONS + 3):
        g.check("file:///x")
    return len(g.drain()) == E.MAX_PENDING_DECISIONS and g.counts["dropped"] == 3


def p_denied_hosts(E):
    g = guard(E)
    g.check("http://x.example.test/a?q=1")
    g.check("http://x.example.test/b")
    return g.denied_hosts() == (("x.example.test", 2),)


def p_record_has_no_path(E):
    g = guard(E, *hosts(E, "app.example.test"))
    d = g.check("http://app.example.test/secret/path?token=abc")
    return "secret" not in repr(d.to_json()) and "abc" not in repr(d.to_json())


def p_policy_id_follows_entries(E):
    a = E.EgressPolicy(entries=tuple(hosts(E, "app.example.test")))
    b = E.EgressPolicy(entries=tuple(hosts(E, "app.example.test", "b.example.test")))
    return a.policy_id != b.policy_id and a.policy_id.startswith("rcv1:") and E.EgressPolicy(entries=a.entries, max_redirects=2).policy_id != a.policy_id


def p_wss_same_origin(E):
    g = guard(E, E.AllowEntry(schemes=("https",), ports=(443,), host="app.example.test"))
    return g.check("wss://app.example.test/").allowed and not g.check("ws://app.example.test:443/").allowed


def p_ipv6_text(E):
    import ipaddress
    return E.address_text(ipaddress.ip_address("::ffff:127.0.0.1")) == "0000:0000:0000:0000:0000:ffff:7f00:0001"


def p_host_port_structure(E):
    g = guard(E, *hosts(E, "app.example.test"))
    pol = g.policy
    return not g.check_host_port("app.example.test/x", 443).allowed and not g.check_host_port("evil.test@app.example.test", 443).allowed \
        and not E.decide_host_port(pol, "app.example.test#y", 443, [PUB]).allowed and g.check_host_port("app.example.test", 443).allowed


def p_host_port_port(E):
    g = guard(E, *hosts(E, "app.example.test"))
    return all(not g.check_host_port("app.example.test", p).allowed for p in (True, "443", 0, 70000, None))


EGRESS = "egress.py"
MUTATIONS = {
    "egress: metadata addresses are not refused when listed":
        (EGRESS, [("        if any(c is AddressClass.METADATA or c is AddressClass.LINK_LOCAL for c in classes.values()):", "        if False:")],
         p_metadata_refused_even_if_listed, "metadata refusal"),
    "egress: metadata hidden in a mapped, NAT64 or 6to4 address is not found":
        (EGRESS, [("    inner = _embedded_v4(addr)\n    return inner is not None and _in_metadata(inner)", "    return False")],
         p_embedded_metadata, "metadata refusal embedded"),
    "egress: metadata names are not refused":
        (EGRESS, [("    if name in METADATA_NAMES:\n        return Reason.DENY_METADATA_NAME", "    if False:\n        return Reason.DENY_METADATA_NAME")],
         p_metadata_names, "metadata refusal names"),
    "egress: internal-suffix names are not refused":
        (EGRESS, [("    if any(name.endswith(suffix) or (name == suffix[1:] and name != \"localhost\") for suffix in INTERNAL_SUFFIXES):\n        return Reason.DENY_INTERNAL_NAME",
                   "    if False:\n        return Reason.DENY_INTERNAL_NAME")],
         p_internal_names, "internal names"),
    "egress: no pinned address in an allowed decision":
        (EGRESS, [("                    pinned_address=address_text(ordered[0]) if ordered else None,", "                    pinned_address=None,")],
         p_pin, "resolve-then-pin"),
    "egress: the peer address is not compared with the pin":
        (EGRESS, [("        return address_text(peer) == self.pinned_address", "        return True")],
         p_peer, "resolve-then-pin peer"),
    "egress: verify_peer accepts any peer":
        (EGRESS, [("        if decision.allowed and decision.permits_peer(peer_address):\n            return decision", "        if decision.allowed:\n            return decision")],
         p_verify_peer, "resolve-then-pin peer check"),
    "egress: redirect hops after the first are not decided again (chain)":
        (EGRESS, [("        decision = decide(policy, absolute, addresses, purpose=purpose if index == 0 else \"redirect\", hop=index,\n                          previous_scheme=previous_scheme)",
                   "        decision = decide(policy, absolute, addresses, purpose=purpose if index == 0 else \"redirect\", hop=index,\n                          previous_scheme=previous_scheme) if index == 0 else decisions[0]")],
         p_redirect_chain, "redirect re-check chain"),
    "egress: a redirect to an unlisted host is not refused (chain)":
        (EGRESS, [("        if not matching:\n            return _deny(pid, [Reason.DENY_NOT_LISTED], purpose, hop, parsed, addresses=ordered,\n                         classes=classes.values(), resolved=resolved)",
                   "        if not matching and hop == 0:\n            return _deny(pid, [Reason.DENY_NOT_LISTED], purpose, hop, parsed, addresses=ordered,\n                         classes=classes.values(), resolved=resolved)\n        if not matching:\n            matching = [(0, AllowEntry(schemes=(parsed.scheme,), ports=(parsed.port,), host=\"app.example.test\"))]")],
         p_redirect_chain_unlisted, "redirect re-check default deny"),
    "egress: the guard does not decide a redirect hop":
        (EGRESS, [("        return self.check(absolute, purpose=\"redirect\", hop=hop,\n                          previous_scheme=previous.scheme if previous is not None else None)",
                   "        return previous if previous is not None else self.check(absolute, purpose=\"redirect\", hop=hop)")],
         p_redirect_guard, "redirect re-check guard"),
    "egress: no hop cap in a chain":
        (EGRESS, [("        if index > policy.max_redirects:", "        if False:")], p_hop_cap_chain, "hop cap chain"),
    "egress: no hop cap in the guard":
        (EGRESS, [("        if hop > self._policy.max_redirects:", "        if False:")], p_hop_cap_guard, "hop cap guard"),
    "egress: a downgrade is not refused (chain)":
        (EGRESS, [("        if previous_scheme is not None and _downgrade(previous_scheme, parsed.scheme):", "        if False:")],
         p_downgrade, "scheme downgrade"),
    "egress: the guard forgets the previous scheme":
        (EGRESS, [("                          previous_scheme=previous.scheme if previous is not None else None)", "                          previous_scheme=None)")],
         p_downgrade_guard, "scheme downgrade guard"),
    "egress: userinfo is not refused":
        (EGRESS, [("    if \"@\" in authority:\n        raise Refused(Reason.DENY_USERINFO)", "    if False:\n        raise Refused(Reason.DENY_USERINFO)")],
         p_userinfo, "userinfo refusal"),
    "egress: an unlisted host is allowed":
        (EGRESS, [("        if not matching:\n            return _deny(pid, [Reason.DENY_NOT_LISTED], purpose, hop, parsed, addresses=ordered,\n                         classes=classes.values(), resolved=resolved)",
                   "        if not matching:\n            return EgressDecision(verdict=EgressVerdict.ALLOW, reasons=(\"ALLOWED_PUBLIC\",), purpose=purpose, hop=hop, policy_id=pid)")],
         p_default_deny, "default deny"),
    "egress: a private entry admits addresses outside its networks":
        (EGRESS, [("            elif not any(addr in n for n in nets if n.version == addr.version):\n                problems.append(Reason.DENY_ADDRESS_OUTSIDE_ENTRY)",
                   "            elif False:\n                problems.append(Reason.DENY_ADDRESS_OUTSIDE_ENTRY)")],
         p_private_networks, "private entry networks"),
    "egress: a private entry admits nothing":
        (EGRESS, [("            elif not any(addr in n for n in nets if n.version == addr.version):\n                problems.append(Reason.DENY_ADDRESS_OUTSIDE_ENTRY)",
                   "            elif True:\n                problems.append(Reason.DENY_ADDRESS_OUTSIDE_ENTRY)")],
         p_private_ok_when_inside, "private entry works"),
    "egress: a public entry admits a non-public address":
        (EGRESS, [("        if cls is not AddressClass.PUBLIC:\n            problems.append", "        if False:\n            problems.append")],
         p_public_entry_private_address, "public entry address rule"),
    "egress: only the first resolved address is checked":
        (EGRESS, [("    for addr in ordered:\n        cls = classes[addr]\n        if cls is not AddressClass.PUBLIC:", "    for addr in ordered[:1]:\n        cls = classes[addr]\n        if cls is not AddressClass.PUBLIC:")],
         p_split_answer, "every address must pass"),
    "egress: IPv4-mapped addresses are not classed as mapped":
        (EGRESS, [("    if addr in _V6_MAPPED or addr in _V6_COMPAT and int(addr) > 1:\n        return AddressClass.MAPPED", "    if False:\n        return AddressClass.MAPPED")],
         p_mapped, "mapped refusal"),
    "egress: a wildcard matches the apex and suffix tricks":
        (EGRESS, [("target.host.endswith(self.host[1:])   # subdomains, never the apex", "target.host.endswith(self.host[2:])")],
         p_wildcard, "wildcard subdomains only"),
    "egress: octal IPv4 spellings are read as decimal":
        (EGRESS, [("        elif len(part) > 1 and part[0] == \"0\":\n            if any(c in \"89\" for c in part):\n                raise Refused(Reason.DENY_HOST_SYNTAX)\n            numbers.append(int(part, 8))",
                   "        elif False:\n            numbers.append(int(part, 8))")],
         p_numeric_octal, "numeric forms octal"),
    "egress: hex IPv4 spellings are not parsed":
        (EGRESS, [("        if part[:2] in (\"0x\", \"0X\"):", "        if False:")], p_numeric_hex, "numeric forms hex"),
    "egress: short and decimal IPv4 spellings are not filled":
        (EGRESS, [("    value = last\n    for i, n in enumerate(head):\n        value += n * 256 ** (3 - i)", "    value = last if len(head) == 0 else 0\n    for i, n in enumerate(head):\n        value += n * 256 ** (3 - i)")],
         p_numeric_decimal_short, "numeric forms short"),
    "egress: an out-of-range numeric host is read as a name":
        (EGRESS, [("    if any(n > 255 for n in head) or last >= 256 ** (5 - len(numbers)):\n        raise Refused(Reason.DENY_HOST_SYNTAX)", "    if False:\n        raise Refused(Reason.DENY_HOST_SYNTAX)")],
         p_numeric_range, "numeric range"),
    "egress: other schemes are not refused":
        (EGRESS, [("    if scheme not in SCHEMES:\n        raise Refused(Reason.DENY_SCHEME)", "    if False:\n        raise Refused(Reason.DENY_SCHEME)")],
         p_scheme, "scheme refusal"),
    "egress: port 0 and above 65535 are not refused":
        (EGRESS, [("        if not 1 <= port <= 65535:\n            raise Refused(Reason.DENY_PORT)", "        if False:\n            raise Refused(Reason.DENY_PORT)")],
         p_port, "port rules"),
    "egress: an unlisted name is resolved":
        (EGRESS, [("        return any(e.matches(target) for e in self._policy.entries)", "        return True")],
         p_no_dns_for_unlisted, "no lookup for unlisted"),
    "egress: IDNA hosts are not converted":
        (EGRESS, [("    if not host.isascii():\n        try:", "    if False:\n        try:")], p_idna, "idna"),
    "egress: percent-encoded hosts are not decoded":
        (EGRESS, [("    if \"%\" in host:\n        try:", "    if False:\n        try:")], p_percent, "percent decoding"),
    "egress: compatibility forms are not folded (nor mapped by IDNA)":
        (EGRESS, [("    folded = unicodedata.normalize(\"NFKC\", host)", "    folded = host"),
                  ("            host = host.encode(\"idna\").decode(\"ascii\")", "            host = host.encode(\"ascii\").decode(\"ascii\")")], p_nfkc, "nfkc"),
    "egress: a trailing dot is kept":
        (EGRESS, [("    if host.endswith(\".\"):\n        host, odd = host[:-1], True", "    if False:\n        host, odd = host[:-1], True")],
         p_trailing_dot, "trailing dot"),
    "egress: an operator entry may be as broad as the whole internet":
        (EGRESS, [("    if net.prefixlen < minimum:", "    if False:")], p_private_entry_breadth, "entry breadth"),
    "egress: a guard without a resolver resolves names":
        (EGRESS, [("        raise LookupError(\"no resolver is configured\")", "        return [\"93.184.216.34\"]")],
         p_no_resolver_for_literals, "no default resolver"),
    "egress: a malformed resolver answer is accepted":
        (EGRESS, [("    if type(item) is not str or not item or item != item.strip() or \"%\" in item:\n        raise Refused(Reason.DENY_RESOLVER_BAD_ANSWER)\n    try:\n        return ipaddress.ip_address(item)\n    except ValueError:\n        raise Refused(Reason.DENY_RESOLVER_BAD_ANSWER) from None",
                   "    return ipaddress.ip_address(\"93.184.216.34\")")],
         p_resolver_bad_answer, "resolver answers"),
    "egress: the decision log is unbounded":
        (EGRESS, [("        if len(self._pending) >= MAX_PENDING_DECISIONS:\n            self._dropped += 1\n        else:\n            self._pending.append(decision)",
                   "        self._pending.append(decision)")],
         p_pending_bound, "log bound"),
    "egress: denied hosts are counted by something other than the host":
        (EGRESS, [("            key = decision.host or \"(unparsed)\"", "            key = decision.host or \"(unparsed)\"\n            key = key + \"/\" + str(decision.hop) + str(decision.purpose)")],
         p_denied_hosts, "hosts only"),
    "egress: the record carries the URL":
        (EGRESS, [("        for key in (\"scheme\", \"host\", \"port\", \"host_kind\", \"pinned_address\", \"entry\"):", "        out[\"url\"] = \"http://\" + str(self.host) + \"/secret/path?token=abc\"\n        for key in (\"scheme\", \"host\", \"port\", \"host_kind\", \"pinned_address\", \"entry\"):")],
         p_record_has_no_path, "record hygiene"),
    "egress: the policy id does not follow the entries":
        (EGRESS, [("        return {\"entries\": [e.to_json() for e in self.entries], \"form\": POLICY_FORM,", "        return {\"entries\": [], \"form\": POLICY_FORM,")],
         p_policy_id_follows_entries, "policy id"),
    "egress: wss is judged by its own scheme only":
        (EGRESS, [("        if target.scheme not in self.schemes and _SAME_ORIGIN_ENTRY.get(target.scheme) not in self.schemes:", "        if target.scheme not in self.schemes:")],
         p_wss_same_origin, "websocket rules"),
    "egress: a host holding URL structure is composed into a URL":
        (EGRESS, [("    if type(host) is not str or not host or any(c in _HOST_PORT_FORBIDDEN or ord(c) < 0x20 for c in host):", "    if type(host) is not str or not host:")],
         p_host_port_structure, "host and port structure"),
    "egress: a non-integer port is composed into a URL":
        (EGRESS, [("    if type(port) is not int or not 1 <= port <= 65535:\n        raise Refused(Reason.DENY_PORT)\n    if type(host)", "    port = int(port) if str(port).isdigit() else port\n    if type(host)")],
         p_host_port_port, "host and port port"),
    "egress: IPv6 text depends on the Python version":
        (EGRESS, [("    n = int(addr)                       # not ``exploded``: Python 3.13 prints a mapped address with a dotted tail\n    return \":\".join(\"%04x\" % ((n >> shift) & 0xFFFF) for shift in range(112, -1, -16))",
                   "    return addr.exploded")],
         p_ipv6_text, "stable text"),
}


# =====================================================================================================================
# probes: redaction
# =====================================================================================================================

def rd(D, **kw):
    kw.setdefault("denylist", H.CANARIES)
    return D.PatternRedactor(D.RedactionConfig(**kw))


def rule_probe(rule_id):
    def probe(D):
        r = rd(D)
        cases = [c for c in RULE_CASES if c[0] == rule_id]
        assert cases
        for _, text, secret in cases:
            out, counts = r.redact_text(text)
            if secret in out or counts.get(rule_id, 0) < 1:
                return False
        return True
    return probe


def p_key_name(D):
    out, _ = rd(D).redact_structure({"password": "x", "auth": "y", "n": "ok"})
    return out == {"password": "[REDACTED:key_name]", "auth": "[REDACTED:key_name]", "n": "ok"}


def p_counters(D):
    out, _ = rd(D).redact_structure({"tokens": 5, "max_tokens": 0, "limits": {"max_tokens": 9}})
    return out == {"tokens": 5, "max_tokens": 0, "limits": {"max_tokens": 9}}


def p_oversize(D):
    r = rd(D, max_input_bytes=10)
    try:
        r.redact(b"z" * 11, "text/plain")
    except RedactionFailed as exc:
        return exc.details["reason"] == "INPUT_TOO_LARGE"
    return False


def p_undecodable(D):
    try:
        rd(D).redact(b"ok \xff\xfe", "text/plain")
    except RedactionFailed as exc:
        return exc.details["reason"] == "NOT_UTF8"
    return False


def p_binary(D):
    try:
        rd(D).redact(b"\x89PNG", "image/png")
    except RedactionFailed:
        return True
    return False


def p_charset(D):
    try:
        rd(D).redact(b"x", "text/plain; charset=latin-1")
    except RedactionFailed:
        return True
    return False


def p_internal_fail_closed(D):
    class Boom(D.PatternRedactor):
        def _text(self, text, counts):
            raise ZeroDivisionError("boom")
    r = Boom(D.RedactionConfig())
    for call in (lambda: r.redact(b"anything", "text/plain"), lambda: r.redact_structure({"a": "b"})):
        try:
            call()
        except RedactionFailed:
            continue
        return False
    return True


SHORT = "qa-ev-x9k2z"          # short enough that its encodings are not caught by the entropy rules


def p_denylist_base64(D):
    import base64
    forms = [base64.b64encode(SHORT.encode()).decode(), base64.b64encode(b"x" + SHORT.encode()).decode()[2:],
             base64.urlsafe_b64encode(SHORT.encode()).decode().rstrip("=")]
    r = rd(D, denylist=(SHORT,))
    return all(f not in r.redact_text(f"v {f} v")[0] for f in forms) and all(len(f) < 24 for f in forms)


def p_denylist_percent(D):
    from urllib.parse import quote
    f = quote(H.CANARY_B, safe="")
    return f not in rd(D).redact_text(f"v {f} v")[0]


def p_denylist_case(D):
    return H.CANARY_A.upper() not in rd(D).redact_text(f"v {H.CANARY_A.upper()} v")[0]


def p_denylist_json_hex(D):
    r = rd(D, denylist=(SHORT,))
    hexed = SHORT.encode().hex()
    return hexed not in r.redact_text(f"v {hexed} v")[0] and len(hexed) < 32 and SHORT not in r.redact_text(json.dumps({"a": SHORT}))[0]


def p_fixed_point(D):
    r = rd(D)
    text = " sk-日本sk-a@b.coAuthorizationa1B2c3a1B2c3a@b.co@"
    once, _ = r.redact_text(text)
    twice, counts = r.redact_text(once)
    return twice == once and counts == {}


def p_digest_exempt(D):
    d = "rcv1:" + "0123456789abcdef" * 4
    r = rd(D)
    return r.redact_text(f"id@{d}")[0] == f"id@{d}" and r.redact_text(d)[0] == d


def p_pii_flags(D):
    text = f"mail {H.EMAIL} call {H.PHONE}"
    return rd(D, pii_email=False, pii_phone=False).redact_text(text)[0] == text and "[REDACTED:pii_email]" in rd(D).redact_text(text)[0]


def p_key_collision(D):
    try:
        rd(D).redact_structure({H.AWS_KEY_ID: 1, "[REDACTED:aws_key_id]": 2})
    except RedactionFailed as exc:
        return exc.details["reason"] == "KEY_COLLISION"
    return False


def p_depth(D):
    deep = cur = {}
    for _ in range(100):
        cur["x"] = {}
        cur = cur["x"]
    try:
        rd(D).redact_structure(deep)
    except RedactionFailed as exc:
        return exc.details["reason"] == "TOO_DEEP"
    return False


def p_luhn(D):
    r = rd(D)
    return r.redact_text("card 4111 1111 1111 1111")[1].get("pii_card") == 1 and "pii_card" not in r.redact_text("card 4111 1111 1111 1112")[1]


def p_json_keys(D):
    out = rd(D).redact(json.dumps({"auth": "plainvalue", "n": 1}).encode(), "application/json")
    return b"plainvalue" not in out


def p_ack_needed(D):
    try:
        rd(D).redact(b"\x00raw", "application/octet-stream")
    except RedactionFailed:
        return rd(D, acknowledged_unredacted_media=("application/octet-stream",)).redact(b"\x00raw", "application/octet-stream") == b"\x00raw"
    return False


def p_policy_rules(D):
    return rd(D).policy_id != rd(D, pii_email=False).policy_id != rd(D, pii_card=False).policy_id


def p_hex_token(D):
    return rd(D).redact_text("key=" + "a1b2c3d4" * 4)[1] == {"hex_token": 1}


def p_entropy_token(D):
    return rd(D).redact_text("id: " + ENT)[1] == {"entropy_token": 1}


def p_nonfinite(D):
    try:
        rd(D).redact_structure([float("nan")])
    except RedactionFailed:
        return True
    return False


def p_unsettled(D):
    class Flip(D.PatternRedactor):
        n = 0

        def _pass(self, text, counts):
            Flip.n += 1
            return text + "x"
    try:
        Flip(D.RedactionConfig())._text("a", {})
    except RedactionFailed as exc:
        return exc.details["reason"] == "UNSTABLE"
    return False


RDN = "redaction.py"
for _rule in [r[0] for r in __import__("agent_harness.qa.redaction", fromlist=["x"]).TEXT_RULES] + ["pii_email", "pii_card", "pii_phone"]:
    MUTATIONS[f"redaction: the {_rule} rule never matches"] = (
        RDN, [(f'("{_rule}", r"', f'("{_rule}", r"(?!)')], rule_probe(_rule), f"rule {_rule}")
MUTATIONS.update({
    "redaction: the key-name rule is off":
        (RDN, [("                    if norm not in exempt and self._key_sensitive(key):", "                    if False:")], p_key_name, "key names"),
    "redaction: counters named tokens are redacted":
        (RDN, [("                        if norm.endswith(\"tokens\") and type(item) in (int, bool):", "                        if False:")], p_counters, "counters"),
    "redaction: no size bound":
        (RDN, [("        if len(data) > self._config.max_input_bytes:", "        if False:")], p_oversize, "fail closed: oversized"),
    "redaction: undecodable bytes are replaced, not refused":
        (RDN, [("            text = bytes(data).decode(\"utf-8\")", "            text = bytes(data).decode(\"utf-8\", \"replace\")")], p_undecodable, "fail closed: undecodable"),
    "redaction: a media type without a rule is passed through":
        (RDN, [("            raise _fail(\"UNSUPPORTED_MEDIA_TYPE\", \"this media type cannot be redacted\")",
                "            return bytes(data), RedactionManifest(rule_set_hash=self._policy_id, counts=(), input_bytes=len(data), output_bytes=len(data), media_type=shown_type)")],
         p_binary, "fail closed: binary"),
    "redaction: any media type counts as acknowledged":
        (RDN, [("            if base in self._config.acknowledged_unredacted_media:", "            if True:")], p_ack_needed, "acknowledgement"),
    "redaction: a foreign charset is read as UTF-8":
        (RDN, [("        if charset is not None and charset not in _OK_CHARSETS:", "        if False:")], p_charset, "fail closed: charset"),
    "redaction: an internal error returns the raw bytes":
        (RDN, [("        except Exception as exc:\n            raise _fail(\"INTERNAL\", \"redaction failed (%s)\" % type(exc).__name__) from exc\n\n    def redact(",
                "        except Exception:\n            return bytes(data), None\n\n    def redact(")],
         p_internal_fail_closed, "fail closed: internal error"),
    "redaction: encoded canaries (base64) are not matched":
        (RDN, [("    forms.update(_b64_variants(data))", "    pass")], p_denylist_base64, "canary base64"),
    "redaction: encoded canaries (percent) are not matched":
        (RDN, [("    forms = {secret, quote(secret, safe=\"\"), quote_plus(secret), ", "    forms = {secret, ")], p_denylist_percent, "canary percent"),
    "redaction: canary case variants are not matched":
        (RDN, [("                forms.update((form, form.lower(), form.upper()))", "                forms.update((form,))")], p_denylist_case, "canary case"),
    "redaction: encoded canaries (hex, JSON) are not matched":
        (RDN, [(" html.escape(secret, quote=True),\n             \"\".join(\"&#%d;\" % ord(c) for c in secret), \"\".join(\"&#x%x;\" % ord(c) for c in secret), data.hex(),",
                " html.escape(secret, quote=True),\n             \"\".join(\"&#%d;\" % ord(c) for c in secret), \"\".join(\"&#x%x;\" % ord(c) for c in secret),")],
         p_denylist_json_hex, "canary hex"),
    "redaction: one pass only (not idempotent)":
        (RDN, [("        for _ in range(MAX_PASSES):\n            result = self._pass(text, counts)\n            if result == text:\n                return result\n            text = result\n        raise _fail(\"UNSTABLE\", \"the redaction did not settle\")",
                "        return self._pass(text, counts)")], p_fixed_point, "idempotence"),
    "redaction: a text that does not settle is accepted":
        (RDN, [("        raise _fail(\"UNSTABLE\", \"the redaction did not settle\")", "        return text")], p_unsettled, "settling"),
    "redaction: full-width digests are redacted":
        (RDN, [("            if tok is None:                                     # a full-width digest reference: left exactly as it is\n                return m.group(0)",
                "            if tok is None:\n                return token_for(\"hex_token\")")], p_digest_exempt, "digests left alone"),
    "redaction: PII rules ignore their flags":
        (RDN, [("            if getattr(self._config, flag):", "            if True:")], p_pii_flags, "pii flags"),
    "redaction: colliding keys are merged":
        (RDN, [("                    if shown in out:\n                        raise _fail(\"KEY_COLLISION\", \"two keys are equal after redaction\")", "                    if False:\n                        pass")],
         p_key_collision, "key collision"),
    "redaction: no depth bound":
        (RDN, [("            if depth > cfg.max_depth:", "            if False:")], p_depth, "depth bound"),
    "redaction: the card rule skips the Luhn check":
        (RDN, [("                if 13 <= len(digits) <= 19 and _luhn(digits):", "                if 13 <= len(digits) <= 19:")], p_luhn, "luhn"),
    "redaction: JSON bodies are redacted as text only":
        (RDN, [("        if is_json_media_type(base):", "        if False:")], p_json_keys, "structured JSON"),
    "redaction: the policy id ignores the rules":
        (RDN, [("               \"rules\": self._rule_docs, \"rules_version\": REDACTION_RULES_VERSION,", "               \"rules\": [], \"rules_version\": REDACTION_RULES_VERSION,")],
         p_policy_rules, "policy id"),
    "redaction: hex tokens are kept":
        (RDN, [("            return \"hex_token\"", "            return None")], p_hex_token, "hex token"),
    "redaction: entropy tokens are kept":
        (RDN, [("                and _entropy_milli(tok) >= ENTROPY_MIN_MILLI:\n            return \"entropy_token\"", "                and _entropy_milli(tok) >= ENTROPY_MIN_MILLI:\n            return None")],
         p_entropy_token, "entropy token"),
    "redaction: a non-finite number passes":
        (RDN, [("                if not math.isfinite(v):", "                if False:")], p_nonfinite, "non-finite"),
})


# =====================================================================================================================
# probes: blobs, vault, integrity
# =====================================================================================================================

def p_digest_width(Bm):
    return not Bm.is_full_digest("sha256:" + "ab" * 8) and not Bm.is_full_digest("sha256:" + "ab" * 31) and Bm.is_full_digest("sha256:" + "ab" * 32)


def p_sink_artifact_bound(Bm):
    sink = Bm.MemoryBlobSink(Bm.EvidenceBounds(max_artifact_bytes=10))
    try:
        sink.put(b"x" * 11, "text/plain")
    except ResourceLimitExceeded:
        return len(sink) == 0
    return False


def p_sink_run_bound(Bm):
    sink = Bm.MemoryBlobSink(Bm.EvidenceBounds(max_artifact_bytes=10, max_run_bytes=15))
    sink.put(b"a" * 10, "text/plain")
    try:
        sink.put(b"b" * 10, "text/plain")
    except ResourceLimitExceeded:
        return len(sink) == 1
    return False


def p_sink_count_bound(Bm):
    sink = Bm.MemoryBlobSink(Bm.EvidenceBounds(max_artifacts=1))
    sink.put(b"a", "text/plain")
    try:
        sink.put(b"b", "text/plain")
    except ResourceLimitExceeded:
        return True
    return False


def p_sink_verifies(Bm):
    sink = Bm.MemoryBlobSink()
    d = sink.put(b"abc", "text/plain")
    sink._blobs[d] = b"abd"
    try:
        sink.get(d)
    except Bm.BlobCorrupt:
        return True
    return False


def p_retention(Bm):
    p = Bm.RetentionPolicy()
    return p.days_for("PASS") == 14 and p.days_for("PRODUCT_FAIL") == 60 and p.days_for("SOMETHING_NEW") == 60 and p.days_for(None) == 60


def p_manifest_content(Bm):
    class Lax:
        """A store whose reads do not verify (a filesystem store may be one)."""
        def __init__(self):
            self.data = {}
        def exists(self, d):
            return d in self.data
        def size(self, d):
            return len(self.data[d])
        def get(self, d):
            return self.data[d]
    store = Lax()
    digest = Bm.digest_of_bytes(b"abc")
    store.data[digest] = b"abd"
    doc = Bm.ArtifactManifest(entries=(Bm.ArtifactEntry(kind="note", digest=digest, size=3, media_type="text/plain"),)).to_json()
    return Bm.verify_artifact_manifest(doc, store) == (("CONTENT_MISMATCH", 0),)


def p_truncated(Bm):
    return Bm.find_truncated_digests({"a": "sha256:" + "ab" * 8, "b": "rcv1:" + "cd" * 20}) == ("$.a", "$.b")


def vault_with(V, B, *, bounds, **kw):
    sink = B.MemoryBlobSink(B.EvidenceBounds(max_artifact_bytes=10 ** 9, max_run_bytes=10 ** 9, max_artifacts=10 ** 6))
    return V.ArtifactVault(H.make_redactor(), sink, bounds=bounds, **kw), sink


def p_vault_unstable(V):
    class Flip:
        policy_id = "rcv1:" + "5" * 64
        n = 0

        def redact(self, data, media_type):
            Flip.n += 1
            return b"stable" if Flip.n % 2 else b"other"
    vault = V.ArtifactVault(Flip(), B_REAL.MemoryBlobSink())
    try:
        vault.store("note", b"x", "text/plain")
    except RedactionFailed as exc:
        return exc.details["reason"] == "UNSTABLE"
    return False


def p_vault_failure_stores_nothing(V):
    vault = V.ArtifactVault(H.make_redactor(), B_REAL.MemoryBlobSink())
    try:
        vault.store("note", b"\xff raw " + H.AWS_KEY_ID.encode(), "text/plain")
    except RedactionFailed:
        return len(vault._sink) == 0 and len(vault.failures) == 1 and vault.manifest().entries == ()
    return False


def p_vault_raw_bound(V):
    vault, sink = vault_with(V, B_REAL, bounds=B_REAL.EvidenceBounds(max_artifact_bytes=30))
    try:
        vault.store("note", b"a1b2c3d4" * 8, "text/plain")           # 64 bytes in, a 20-byte token out: only the raw bound refuses it
    except ResourceLimitExceeded:
        return len(sink) == 0
    return False


def p_vault_out_bound(V):
    vault, sink = vault_with(V, B_REAL, bounds=B_REAL.EvidenceBounds(max_artifact_bytes=40))
    try:
        vault.store("note", (b"pwd=kkkk " * 4), "text/plain")
    except ResourceLimitExceeded:
        return len(sink) == 0
    return False


def p_vault_run_bound(V):
    vault, sink = vault_with(V, B_REAL, bounds=B_REAL.EvidenceBounds(max_artifact_bytes=100, max_run_bytes=120))
    vault.store("a", b"y" * 60, "text/plain")
    try:
        vault.store("b", b"x" * 80, "text/plain")
    except ResourceLimitExceeded:
        return len(sink) == 1
    return False


def p_vault_count_bound(V):
    vault, sink = vault_with(V, B_REAL, bounds=B_REAL.EvidenceBounds(max_artifacts=1))
    vault.store("a", b"y", "text/plain")
    try:
        vault.store("b", b"x", "text/plain")
    except ResourceLimitExceeded:
        return True
    return False


def p_vault_digest_check(V):
    class Lying(B_REAL.MemoryBlobSink):
        def put(self, data, media_type):
            super().put(data, media_type)
            return "sha256:" + "0" * 64
    try:
        V.ArtifactVault(H.make_redactor(), Lying()).store("n", b"fine", "text/plain")
    except B_REAL.BlobRejected:
        return True
    return False


def p_vault_retention(V):
    vault = V.ArtifactVault(H.make_redactor(), B_REAL.MemoryBlobSink(), retention=B_REAL.RetentionPolicy(passing_days=3, failing_days=9))
    vault.store("n", b"x", "text/plain")
    return vault.manifest("PASS").retention_days == 3 and vault.manifest("INCONCLUSIVE").retention_days == 9


def g(V, **bounds):
    return V.EvidenceGuard(H.make_redactor(), B_REAL.EvidenceBounds(**bounds) if bounds else None)


def apply(guard, kind, payload, record=None):
    return guard.apply(kind, record, payload, "canon-v1:" + "0" * 64, seq=0)


def p_guard_redacts(V):
    out, _ = apply(g(V), "X", {"a": H.AWS_KEY_ID, "password": "p"})
    return H.AWS_KEY_ID not in json.dumps(out) and out["password"] == "[REDACTED:key_name]"


def p_guard_exempt(V):
    out, _ = apply(g(V), "X", {"secret_ref": {"provider": "v", "key": "k"}})
    return out == {"secret_ref": {"provider": "v", "key": "k"}}


def p_guard_event_bound(V):
    try:
        apply(g(V, max_event_bytes=50), "X", {"a": "x" * 100})
    except ResourceLimitExceeded as exc:
        return exc.details["bound"] == "event"
    return False


def p_guard_count_bound(V):
    guard = g(V, max_events=1)
    apply(guard, "X", {"a": 1})
    try:
        apply(guard, "X", {"a": 2})
    except ResourceLimitExceeded as exc:
        return exc.details["bound"] == "events"
    return False


def p_guard_total_bound(V):
    guard = g(V, max_events_bytes=50)
    apply(guard, "X", {"a": "x" * 20})
    try:
        apply(guard, "X", {"a": "y" * 40})
    except ResourceLimitExceeded as exc:
        return exc.details["bound"] == "events_bytes"
    return False


def p_guard_seal_reserve(V):
    guard = g(V, max_events=1, max_events_bytes=50, seal_reserve_bytes=100)
    apply(guard, "X", {"a": "x"})
    apply(guard, "RunSealed", {"a": "y" * 30})
    try:
        apply(guard, "RunSealed", {"a": "z" * 100})
    except ResourceLimitExceeded as exc:
        return exc.details["bound"] == "seal"
    return False


def p_guard_refs(V):
    try:
        apply(g(V), "Observation", {"screenshot_ref": {"ref": "sha256:" + "ab" * 8, "content_hash": "sha256:" + "ab" * 8}})
    except B_REAL.BlobRejected:
        return True
    return False


def p_guard_record_rebuild(V):
    guard = g(V)
    obs = R.Observation(observation_id="o-1", step_id="s-1", phase=R.Phase.BEFORE,
                        source=R.ObservationSource(adapter="a", adapter_version="1", browser="c", browser_version="1"),
                        redaction_applied=True, redaction_policy_id=guard.policy_id, url=f"https://x.test/p?token={H.TOK_A}",
                        dom_meta={"title": H.CANARY_A})
    out, h = guard.apply("Observation", obs, obs.to_json(with_hash=True), obs.canon_hash(), seq=0)
    return R.Observation.from_json(out).canon_hash() == h and h != obs.canon_hash()


def p_guard_tracks_changes(V):
    guard = g(V)
    apply(guard, "X", {"a": H.AWS_KEY_ID})
    return guard.redacted_events == ((0, "X"),) and guard.manifest_payload()["tally"]["counts"] == {"aws_key_id": 1}


def p_guard_failure_payload(V):
    p = g(V).failure_payload("Observation", RedactionFailed("secret " + H.AWS_KEY_ID, reason="NOT_UTF8"))
    return p == {"code": "REDACTION_FAILED", "form": "qa-evidence-failure/1", "kind": "Observation", "reason": "NOT_UTF8"}


# --- integrity ------------------------------------------------------------------------------------------------------

def a_run():
    env = H.evidence_env()
    r = env.run()
    return r


_RUN = []


def run_once():
    if not _RUN:
        _RUN.append(a_run())
    return _RUN[0]


def check(I, **kw):
    r = run_once()
    base = dict(events=r.events, receipts=r.receipts, links=r.receipt_links, manifest=r.manifest)
    base.update(kw)
    return I.verify_run_evidence(**base)


def p_int_clean(I):
    return check(I).ok


def p_int_event_chain(I):
    events = list(run_once().events)
    return "EVENT_CHAIN" in check(I, events=events[:3] + events[4:]).codes


def p_int_receipt_chain(I):
    r = run_once()
    return "RECEIPT_CHAIN" in check(I, receipts=r.receipts[::-1]).codes


def p_int_count(I):
    r = run_once()

    def drop(i, kind, payload):
        return None if kind == "ActionReceipt" and payload["step_id"] == "s-save" else (kind, payload)
    from .test_evidence_integrity import rechain
    return "RECEIPT_EVENT_COUNT" in check(I, events=rechain(r.events, drop)).codes


def p_int_mismatch(I):
    from .test_evidence_integrity import rechain
    r = run_once()

    def tweak(i, kind, payload):
        if kind == "ActionReceipt" and payload["step_id"] == "s-save":
            payload["result"] = "ACTION_TIMEOUT"
        return kind, payload
    return codes_of(check(I, events=rechain(r.events, tweak))) == ["RECEIPT_EVENT_MISMATCH"]


def codes_of(report):
    return sorted(set(report.codes))


def p_int_redacted_exemption(I):
    from .test_evidence_integrity import rechain
    r = run_once()
    save = next(i for i, e in enumerate(r.events) if e["kind"] == "ActionReceipt" and e["payload"]["step_id"] == "s-save")

    def listed(i, kind, payload):
        if i == save:
            payload["marker"] = "x"
        if kind == "RedactionManifest":
            payload["redacted_events"] = payload["redacted_events"] + [{"kind": "ActionReceipt", "seq": save}]
        return kind, payload
    return check(I, events=rechain(r.events, listed)).ok


def p_int_seal_head(I):
    from .test_evidence_integrity import rechain

    def head(i, kind, payload):
        if kind == "RunSealed":
            payload["receipt_head"] = "canon-v1:" + "3" * 64
        return kind, payload
    return "SEAL_RECEIPT_HEAD" in check(I, events=rechain(run_once().events, head)).codes


def p_int_seal_manifest(I):
    import dataclasses
    other = dataclasses.replace(run_once().manifest, outcome=R.Outcome.INCONCLUSIVE, outcome_reason="X")
    return "SEAL_MANIFEST_HASH" in check(I, manifest=other).codes


def p_int_not_sealed(I):
    return "NOT_SEALED" in check(I, events=list(run_once().events)[:-1]).codes


def p_int_refs(I):
    from .test_evidence_integrity import rechain, with_ref
    short = "sha256:" + "ab" * 8
    return "BAD_EVIDENCE_REF" in check(I, events=rechain(run_once().events, with_ref({"ref": short, "content_hash": short}))).codes


def p_int_truncated(I):
    from .test_evidence_integrity import rechain

    def mutate(i, kind, payload):
        if kind == "RunStarted":
            payload["note"] = "ref sha256:" + "ab" * 8
        return kind, payload
    return "TRUNCATED_DIGEST" in check(I, events=rechain(run_once().events, mutate)).codes


def p_int_malformed(I):
    return "EVENT_CHAIN" in I.verify_run_evidence(events=[None]).codes


def p_int_blob_missing(I):
    from .test_evidence_integrity import rechain, with_ref
    sink = B_REAL.MemoryBlobSink()
    digest = sink.put(b"png", "text/plain")
    events = rechain(run_once().events, with_ref(B_REAL.evidence_ref(digest, "text/plain").to_json()))
    return check(I, events=events, blobs=sink).ok and "MISSING_BLOB" in check(I, events=events, blobs=B_REAL.MemoryBlobSink()).codes


def p_int_artifacts(I):
    sink = B_REAL.MemoryBlobSink()
    d = sink.put(b"abc", "text/plain")
    m = B_REAL.ArtifactManifest(entries=(B_REAL.ArtifactEntry(kind="n", digest=d, size=3, media_type="text/plain"),))
    sink._blobs[d] = b"abd"
    return any(c.startswith("ARTIFACT_") for c in check(I, artifacts=m, blobs=sink).codes) and "ARTIFACTS_UNCHECKED" in check(I, artifacts=m).codes


BLB, VLT, INT = "blobs.py", "vault.py", "integrity.py"
MUTATIONS.update({
    "blobs: short digests are accepted":
        (BLB, [('_FULL_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")', '_FULL_DIGEST = re.compile(r"sha256:[0-9a-f]{16,64}")')], p_digest_width, "full-width digests"),
    "blobs: no per-artifact bound":
        (BLB, [("        if len(data) > limit:", "        if False:")], p_sink_artifact_bound, "artifact bound"),
    "blobs: no run bound":
        (BLB, [("        if self._total + len(data) > self._bounds.max_run_bytes:", "        if False:")], p_sink_run_bound, "run bound"),
    "blobs: no artifact count bound":
        (BLB, [("        if len(self._blobs) >= self._bounds.max_artifacts:", "        if False:")], p_sink_count_bound, "count bound"),
    "blobs: reads are not verified":
        (BLB, [("        if digest_of_bytes(data) != digest:\n            raise BlobCorrupt(\"a stored blob does not hash to its address\")", "        if False:\n            raise BlobCorrupt(\"x\")")],
         p_sink_verifies, "content addressing"),
    "blobs: unknown outcomes get the short retention":
        (BLB, [("        return RETENTION_PASSING if outcome == \"PASS\" else RETENTION_FAILING", "        return RETENTION_FAILING if outcome in (\"PRODUCT_FAIL\", \"TEST_DEFECT\", \"ENVIRONMENT_FAIL\", \"INCONCLUSIVE\") else RETENTION_PASSING")],
         p_retention, "retention"),
    "blobs: artifact content is not re-hashed":
        (BLB, [("        if digest_of_bytes(data) != digest:\n            findings.append((\"CONTENT_MISMATCH\", i))", "        if False:\n            findings.append((\"CONTENT_MISMATCH\", i))")],
         p_manifest_content, "artifact hashes"),
    "blobs: truncated digests are not found":
        (BLB, [("[0-9a-f]{1,63}(?![0-9a-f])", "[0-9a-f]{1,3}(?![0-9a-f])")], p_truncated, "truncated digest scan"),
    "vault: the second-pass stability check is removed":
        (VLT, [("            if self._redactor.redact(out, media_type) != out:           # a second pass must change nothing", "            if False:")], p_vault_unstable, "stable redaction"),
    "vault: a redaction failure stores the raw bytes":
        (VLT, [("            out, manifest = self._redacted(bytes(data), media_type)\n            if len(out) > limit:",
                "            try:\n                out, manifest = self._redacted(bytes(data), media_type)\n            except RedactionFailed:\n                out, manifest = bytes(data), None\n            if len(out) > limit:")],
         p_vault_failure_stores_nothing, "fail closed: store nothing"),
    "vault: no bound on the raw input":
        (VLT, [("            if size > limit:\n                raise ResourceLimitExceeded(\"the artifact is over its size bound\"", "            if False:\n                raise ResourceLimitExceeded(\"the artifact is over its size bound\"")],
         p_vault_raw_bound, "raw size bound"),
    "vault: no bound on the redacted output":
        (VLT, [("            if len(out) > limit:", "            if False:")], p_vault_out_bound, "output size bound"),
    "vault: no run bound":
        (VLT, [("            if self._total + len(out) > self._bounds.max_run_bytes:", "            if False:")], p_vault_run_bound, "run size bound"),
    "vault: no count bound":
        (VLT, [("            if len(self._entries) >= self._bounds.max_artifacts:", "            if False:")], p_vault_count_bound, "artifact count bound"),
    "vault: the sink's digest is not checked":
        (VLT, [("            if digest != digest_of_bytes(out):", "            if False:")], p_vault_digest_check, "content address check"),
    "vault: the manifest ignores the outcome":
        (VLT, [("retention_class=self._retention.retention_class(outcome),\n                                retention_days=self._retention.days_for(outcome)", "retention_class=self._retention.retention_class(None),\n                                retention_days=self._retention.days_for(None)")],
         p_vault_retention, "retention in the manifest"),
    "guard: events are not redacted":
        (VLT, [("        new_payload, counts = self._redactor.redact_structure(payload, exempt_keys=self._exempt)", "        new_payload, counts = payload, {}")],
         p_guard_redacts, "events redacted"),
    "guard: structural keys are redacted":
        (VLT, [("exempt_keys=self._exempt)", "exempt_keys=())")], p_guard_exempt, "structural keys"),
    "guard: no per-event bound":
        (VLT, [("        if not seal and size > self._bounds.max_event_bytes:", "        if False:")], p_guard_event_bound, "event size bound"),
    "guard: no event count bound":
        (VLT, [("        if not seal and self._events >= self._bounds.max_events:", "        if False:")], p_guard_count_bound, "event count bound"),
    "guard: no total size bound":
        (VLT, [("            if self._bytes + size > self._bounds.max_events_bytes:", "            if False:")], p_guard_total_bound, "events total bound"),
    "guard: closing events are unbounded":
        (VLT, [("            if self._seal_bytes + size > self._bounds.seal_reserve_bytes:", "            if False:")], p_guard_seal_reserve, "seal reserve"),
    "guard: references are not checked":
        (VLT, [("                require_blob_ref(value)                  # a short or foreign hash in an evidence reference is refused", "                pass")],
         p_guard_refs, "full-width references"),
    "guard: a redacted record keeps its old hash":
        (VLT, [("                    new_payload, payload_hash = rebuilt.to_json(with_hash=True), rebuilt.canon_hash()", "                    new_payload = rebuilt.to_json(with_hash=True)")],
         p_guard_record_rebuild, "record rebuild"),
    "guard: changed events are not listed":
        (VLT, [("            self._redacted.append((seq, kind))", "            pass")], p_guard_tracks_changes, "changed-event list"),
    "guard: the failure marker carries the reason text":
        (VLT, [("\"reason\": _reason_of(exc)}", "\"reason\": _reason_of(exc), \"message\": str(exc)}")], p_guard_failure_payload, "failure marker hygiene"),
    "integrity: the event chain is not verified":
        (INT, [("        event_head = verify_events(list(events))", "        event_head = None")], p_int_event_chain, "event chain"),
    "integrity: the receipt chain is not verified":
        (INT, [("            receipt_head = verify_chain(list(receipts), list(links))", "            receipt_head = None")], p_int_receipt_chain, "receipt chain"),
    "integrity: the receipt count is not compared":
        (INT, [("        if len(stream) != len(receipts):", "        if False:")], p_int_count, "stream and ledger count"),
    "integrity: a receipt event may differ from the ledger":
        (INT, [("                if not same and seq not in redacted:", "                if False:")], p_int_mismatch, "stream and ledger identity"),
    "integrity: a redacted receipt event is held to the ledger":
        (INT, [("                if not same and seq not in redacted:", "                if not same:")], p_int_redacted_exemption, "redaction exemption"),
    "integrity: the seal's receipt head is not checked":
        (INT, [("        if links is not None and payload.get(\"receipt_head\") != (links[-1].link_hash if links else None):", "        if False:")],
         p_int_seal_head, "seal head"),
    "integrity: the seal's manifest hash is not checked":
        (INT, [("        if manifest is not None and payload.get(\"manifest_hash\") != manifest.canon_hash():", "        if False:")], p_int_seal_manifest, "seal manifest"),
    "integrity: an unsealed run is not reported":
        (INT, [("        if require_sealed:\n            findings.append(Finding(\"NOT_SEALED\"))", "        if False:\n            findings.append(Finding(\"NOT_SEALED\"))")],
         p_int_not_sealed, "sealed"),
    "integrity: evidence references are not checked":
        (INT, [("                if not is_full_digest(digest) or ref.get(\"ref\") != digest:", "                if False:")], p_int_refs, "references"),
    "integrity: truncated digests are not looked for":
        (INT, [("    for where in find_truncated_digests([e.get(\"payload\") if isinstance(e, Mapping) else None for e in events]):", "    for where in ():")],
         p_int_truncated, "truncated digests"),
    "integrity: malformed events are accepted":
        (INT, [("    if not isinstance(events, (list, tuple)) or not all(isinstance(e, Mapping) for e in events):", "    if False:")], p_int_malformed, "malformed events"),
    "integrity: missing blobs are not found":
        (INT, [("                        if not blobs.exists(digest):", "                        if False:"),
               ("                    except QAError:\n                        findings.append(Finding(\"MISSING_BLOB\", f\"{seq}.{name}\"))", "                    except QAError:\n                        pass")],
         p_int_blob_missing, "blobs exist"),
    "integrity: artifacts are not verified":
        (INT, [("            for code, index in verify_artifact_manifest(artifacts, blobs):", "            for code, index in ():")], p_int_artifacts, "artifact hashes"),
})


# =====================================================================================================================
# probes: the runner's wiring
# =====================================================================================================================

def env_for(RN, *, spec=None, egress=None, bounds=None, adapter_setup=None, redactor=None, config_extra=None, **kw):
    guard = EvidenceGuard(redactor or H.make_redactor(), bounds)
    if egress is None:
        egress, _ = H.make_guard(H.app_policy(), H.APP_TABLE)
    env = make_env(spec, mod=RN, config_kw={"egress_policy_id": egress.policy_id, "redaction_policy_id": guard.policy_id,
                                            **(config_extra or {})},
                   evidence_guard=guard, egress=egress, adapter_setup=adapter_setup, **kw)
    env.guard, env.egress = guard, egress
    return env


def metadata_env(RN):
    spec = make_spec([nav_step(url="http://169.254.169.254/latest"), save_step()])
    return env_for(RN, spec=spec, policy=default_policy(spec, allowed_origins=[APP, "http://169.254.169.254"]))


def p_run_precheck(RN):
    env = metadata_env(RN)
    r = env.run()
    return operations(env.adapter, "act") == [] and r.outcome is R.Outcome.ENVIRONMENT_FAIL and r.receipts[0].result is R.ResultCode.NAVIGATION_DENIED


def p_run_link(RN):
    env = metadata_env(RN)
    r = env.run()
    events = [e["payload"] for e in r.events if e["kind"] == "EgressDecision" and e["payload"]["decision"]["verdict"] == "DENY"]
    return len(events) == 1 and events[0]["receipt_ref"] == RC.receipt_ref(r.receipts[0]) and events[0]["step_id"] == "s-nav"


def p_run_summaries(RN):
    r = env_for(RN).run()
    kinds = [e["kind"] for e in r.events]
    return r.outcome is R.Outcome.PASS and kinds[-3:] == ["EgressSummary", "RedactionManifest", "RunSealed"]


def p_run_guard_applied(RN):
    def setup(adapter):
        adapter.dom_meta = {"title": f"x {H.CANARY_A}", "password": H.PW_A, "nodes": 1}
    env = env_for(RN, adapter_setup=setup)
    r = env.run()
    text = json.dumps([e["payload"] for e in r.events], sort_keys=True)
    return H.CANARY_A not in text and H.PW_A not in text and r.sealed


def p_run_failure_marker(RN):
    class Poisoned(H.RD.PatternRedactor):
        def redact_structure(self, value, *, exempt_keys=()):
            if "POISON" in json.dumps(value, default=str):
                raise RedactionFailed("x", reason="NOT_UTF8")
            return super().redact_structure(value, exempt_keys=exempt_keys)

    def setup(adapter):
        adapter.dom_meta = {"title": "POISON"}
    env = env_for(RN, adapter_setup=setup, redactor=Poisoned(H.RD.RedactionConfig()))
    r = env.run()
    kinds = [e["kind"] for e in r.events]
    return "EvidenceFailure" in kinds and r.outcome is R.Outcome.INCONCLUSIVE and r.classification.reason == "REDACTION_FAILED" \
        and "POISON" not in json.dumps([e["payload"] for e in r.events])


def p_run_observed_url(RN):
    def setup(adapter):
        def go(adapter, url):
            adapter.url = "https://evil.example.test/landing"
        adapter.on_navigate = go
    r = env_for(RN, adapter_setup=setup).run()
    return r.outcome is R.Outcome.ENVIRONMENT_FAIL and r.classification.reason == "NAVIGATION_DENIED"


def p_run_seal_tolerance(RN):
    env = env_for(RN, bounds=B_REAL.EvidenceBounds(seal_reserve_bytes=1))
    r = env.run()
    return r.sealed is False and r.outcome is R.Outcome.ENVIRONMENT_FAIL


def p_run_ctor_evidence(RN):
    guard = EvidenceGuard(H.make_redactor())
    try:
        make_env(mod=RN, evidence_guard=guard)
    except SpecInvalid:
        return True
    return False


def p_run_ctor_egress(RN):
    egress, _ = H.make_guard(H.app_policy(), H.APP_TABLE)
    try:
        make_env(mod=RN, egress=egress)
    except SpecInvalid:
        return True
    return False


def p_run_flush_at_step_end(RN):
    from agent_harness.qa.errors import ActionTimeout
    from ._fx_runner import Fault
    from .test_runner import nav_timeout, retry_spec
    egress, _ = H.make_guard(H.app_policy(), H.APP_TABLE)

    def setup(adapter):
        nav_timeout()(adapter)                                           # the first navigation attempt times out
        original = adapter.act
        seen = []

        def act(s, action, target):
            if not seen:
                seen.append(1)
                egress.check("https://beacon.exfil.test/")              # a blocked request during attempt 1
            return original(s, action, target)
        adapter.act = act
    env = env_for(RN, egress=egress, spec=retry_spec(), adapter_setup=setup, config_extra={"recovery_enabled": True})
    r = env.run()
    blocked = [e["payload"] for e in r.events if e["kind"] == "EgressDecision" and e["payload"]["decision"]["verdict"] == "DENY"]
    first = [x for x in r.receipts if x.step_id == "s-nav"][0]
    return len(blocked) == 1 and first.attempt == 1 and blocked[0]["receipt_ref"] == RC.receipt_ref(first) and \
        len([x for x in r.receipts if x.step_id == "s-nav"]) == 2


RUN = "runner.py"
MUTATIONS.update({
    "runner: a navigation is not checked against the allow-list before the adapter":
        (RUN, [("            if not self._egress.check(step.action.value, purpose=\"navigation\").allowed:", "            if False:")], p_run_precheck, "egress pre-check"),
    "runner: egress events do not name the receipt":
        (RUN, [("                                             \"receipt_ref\": receipt_ref(receipt) if receipt is not None else None,", "                                             \"receipt_ref\": None,")],
         p_run_link, "receipt-linked egress events"),
    "runner: the closing evidence events are not written":
        (RUN, [("            self._emit_summaries(c)\n", "")], p_run_summaries, "closing events"),
    "runner: events are not passed through the guard":
        (RUN, [("                payload, phash = self._guard.apply(kind, record, payload, phash, seq=len(c.events))", "                pass")],
         p_run_guard_applied, "redaction before storage"),
    "runner: a failed event leaves no marker":
        (RUN, [("                self._append(c, \"EvidenceFailure\", self._guard.failure_payload(kind, exc), tolerate=True)", "                pass")],
         p_run_failure_marker, "failure marker"),
    "runner: the observed URL is not held to the allow-list":
        (RUN, [("        if where is not None and not where.allowed:", "        if False:")], p_run_observed_url, "observed URL"),
    "runner: a guard failure while sealing escapes the run":
        (RUN, [("        except (EvidenceUnavailable, RedactionFailed, ResourceLimitExceeded, BlobRejected):", "        except EvidenceUnavailable:")],
         p_run_seal_tolerance, "seal tolerance"),
    "runner: the evidence guard's policy id is not checked":
        (RUN, [("                                           or evidence_guard.policy_id != config.redaction_policy_id):", "):")], p_run_ctor_evidence, "redaction policy id check"),
    "runner: the egress guard's policy id is not checked":
        (RUN, [("        if egress is not None and (type(egress) is not EgressGuard or egress.policy_id != config.egress_policy_id):", "        if egress is not None and type(egress) is not EgressGuard:")],
         p_run_ctor_egress, "egress policy id check"),
    "runner: decisions made during a step are not linked to its receipt":
        (RUN, [("        self._flush_egress(c, step.step_id, receipt)\n        return _Attempt(", "        return _Attempt(")], p_run_flush_at_step_end, "attempt flush"),
})


# =====================================================================================================================
# the table
# =====================================================================================================================

def test_every_mutation_anchor_is_unique_and_every_probe_holds_on_the_real_code():
    failed = [name for name, (filename, replacements, probe, why) in MUTATIONS.items()
              if probe(real(filename)) is not True]
    assert failed == []


@pytest.mark.parametrize("name", list(MUTATIONS))
def test_removing_the_guard_makes_the_probe_fail(name, mutants):
    filename, replacements, probe, why = MUTATIONS[name]
    mutant = mutants.load(filename, replacements)
    assert not holds(probe, mutant), f"mutation survived: {name} ({why})"


def test_the_mutation_table_covers_the_guards_the_plan_names():
    touched = {filename for filename, *_ in MUTATIONS.values()}
    assert touched == {"egress.py", "redaction.py", "blobs.py", "vault.py", "integrity.py", "runner.py"}
    wanted = ("metadata refusal", "resolve-then-pin", "redirect re-check", "userinfo refusal", "fail closed: undecodable",
              "fail closed: internal error", "fail closed: store nothing", "artifact bound", "run bound", "full-width digests",
              "default deny", "hop cap", "scheme downgrade", "canary base64", "idempotence", "key names", "event size bound",
              "redaction before storage", "receipt-linked egress events", "stream and ledger", "seal head", "retention")
    assert all(any(w in why for _, _, _, why in MUTATIONS.values()) for w in wanted)
    assert len(MUTATIONS) >= 100

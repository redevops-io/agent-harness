# SPDX-License-Identifier: AGPL-3.0-or-later
"""The egress guard: resolve once and pin, verify the peer, re-decide every connection and every redirect hop, and
log every decision (allow or deny) with hosts only."""
import ipaddress

import pytest

from agent_harness.qa import egress as E

from . import _fx_evidence as H


class Flipping:
    """DNS rebinding: a public answer the first time, a private (or metadata) one afterwards."""

    def __init__(self, first, later):
        self.first, self.later, self.calls = first, later, 0

    def resolve(self, host):
        self.calls += 1
        return self.first if self.calls == 1 else self.later


def policy_for(host="app.example.test", ports=(443,), schemes=("https",)):
    return E.EgressPolicy(entries=(E.AllowEntry(schemes=schemes, ports=ports, host=host),))


def test_the_resolver_is_called_once_and_the_connection_is_pinned_to_the_checked_address():
    resolver = Flipping([H.PUBLIC_V4], ["169.254.169.254"])
    guard = E.EgressGuard(policy_for(), resolver)
    d = guard.check("https://app.example.test/")
    assert d.allowed and resolver.calls == 1
    address, port, server_name = d.connect_target()
    assert (address, port, server_name) == (H.PUBLIC_V4, 443, "app.example.test")
    # the connector uses the pin: a second lookup, which would now say metadata, never happens
    assert resolver.calls == 1


def test_a_rebound_connection_is_caught_by_the_peer_check():
    guard = E.EgressGuard(policy_for(), Flipping([H.PUBLIC_V4], ["10.0.0.5"]))
    d = guard.check("https://app.example.test/")
    assert guard.verify_peer(d, H.PUBLIC_V4) is d
    assert guard.verify_peer(d, ipaddress.ip_address(H.PUBLIC_V4)) is d
    denied = guard.verify_peer(d, "10.0.0.5")
    assert not denied.allowed and denied.reasons == ("DENY_PEER_MISMATCH",) and denied.host == "app.example.test"
    for peer in ("169.254.169.254", "::ffff:93.184.216.34", "not-an-address", "", None, 12345, "93.184.216.35"):
        assert not guard.verify_peer(d, peer).allowed, peer
    assert not d.permits_peer("93.184.216.35")


def test_a_denied_decision_permits_no_peer():
    guard = E.EgressGuard(policy_for(), E.StaticResolver({"app.example.test": ["10.0.0.5"]}))
    d = guard.check("https://app.example.test/")
    assert not d.allowed and not d.permits_peer("10.0.0.5") and not guard.verify_peer(d, "10.0.0.5").allowed


def test_every_new_connection_is_a_new_decision_made_on_a_fresh_resolution():
    resolver = Flipping([H.PUBLIC_V4], ["10.0.0.5"])
    guard = E.EgressGuard(policy_for(), resolver)
    first = guard.check("https://app.example.test/a")
    second = guard.check("https://app.example.test/b")
    assert first.allowed and not second.allowed and resolver.calls == 2
    assert "DENY_ADDRESS_NOT_PUBLIC" in second.reasons


def test_the_pin_is_the_lowest_allowed_address_whatever_order_the_resolver_answers_in():
    for answer in ([H.PUBLIC_V6, H.PUBLIC_V4_B, H.PUBLIC_V4], [H.PUBLIC_V4, H.PUBLIC_V4_B, H.PUBLIC_V6],
                   [H.PUBLIC_V4_B, H.PUBLIC_V4, H.PUBLIC_V6, H.PUBLIC_V4]):
        guard = E.EgressGuard(policy_for(), E.StaticResolver({"app.example.test": answer}))
        d = guard.check("https://app.example.test/")
        assert d.pinned_address == H.PUBLIC_V4 and d.addresses == (H.PUBLIC_V4, H.PUBLIC_V4_B, E.address_text(ipaddress.ip_address(H.PUBLIC_V6)))
    hashes = {E.EgressGuard(policy_for(), E.StaticResolver({"app.example.test": a})).check("https://app.example.test/").record_hash
              for a in ([H.PUBLIC_V4, H.PUBLIC_V4_B], [H.PUBLIC_V4_B, H.PUBLIC_V4])}
    assert len(hashes) == 1


def test_one_bad_answer_among_good_ones_denies_the_whole_set():
    for bad in ("10.0.0.5", "127.0.0.1", "169.254.169.254", "::1", "fd00::1", "::ffff:1.2.3.4"):
        guard = E.EgressGuard(policy_for(), E.StaticResolver({"app.example.test": [H.PUBLIC_V4, bad]}))
        assert not guard.check("https://app.example.test/").allowed, bad


def test_the_log_holds_every_decision_in_order_and_drain_empties_it():
    guard, _ = H.make_guard()
    urls = ["https://app.example.test/", "http://169.254.169.254/", "http://unlisted.example.test/x?secret=1",
            "file:///etc/passwd", "https://app.example.test/"]
    decisions = [guard.check(u) for u in urls]
    drained = guard.drain()
    assert [d.record_hash for d in drained] == [d.record_hash for d in decisions]
    assert [d.allowed for d in drained] == [True, False, False, False, True]
    assert guard.drain() == ()
    assert guard.counts == {"allowed": 2, "denied": 3, "dropped": 0}


def test_denied_hosts_are_counted_by_host_only():
    guard, _ = H.make_guard()
    for u in ("http://unlisted.example.test/a?x=1", "http://unlisted.example.test/b#frag", "http://other.test:81/c",
              "http://169.254.169.254/latest", "file:///x", "http://user:" + H.gen("g-pw", 8) + H.AT + "unlisted.example.test/"):
        guard.check(u)
    got = dict(guard.denied_hosts())
    assert got == {"(unparsed)": 2, "169.254.169.254": 1, "other.test": 1, "unlisted.example.test": 2}
    assert [h for h, _ in guard.denied_hosts()] == sorted(got)
    assert not any("/" in h or "?" in h or "x=1" in h for h in got)


def test_the_pending_log_is_bounded_and_the_overflow_is_counted_not_silent():
    guard, _ = H.make_guard()
    for _ in range(E.MAX_PENDING_DECISIONS + 10):
        guard.check("file:///x")
    assert len(guard.drain()) == E.MAX_PENDING_DECISIONS
    assert guard.counts["dropped"] == 10 and guard.counts["denied"] == E.MAX_PENDING_DECISIONS + 10


def test_check_static_judges_the_listing_without_resolving():
    guard, resolver = H.make_guard()
    d = guard.check_static("https://app.example.test/page")
    assert d.allowed and not d.resolved and "STATIC_ONLY" in d.reasons and d.pinned_address is None
    assert not guard.check_static("https://unlisted.example.test/").allowed
    assert not guard.check_static("http://169.254.169.254/").allowed
    assert not guard.check_static("http://127.0.0.1/").allowed          # a listed literal is still judged by address
    assert resolver.calls == []
    with pytest.raises(PermissionError):
        d.connect_target()                                              # nothing was pinned: there is nothing to connect to


def test_redirect_hops_are_decided_from_scratch_and_capped():
    guard, resolver = H.make_guard(E.EgressPolicy(entries=H.make_policy().entries, max_redirects=2))
    first = guard.check("https://app.example.test/", purpose="navigation")
    hop1 = guard.check_redirect("https://app.example.test/", "https://api.example.test/", hop=1, previous=first)
    hop2 = guard.check_redirect("https://api.example.test/", "/x", hop=2, previous=hop1)
    calls = len(resolver.calls)
    hop3 = guard.check_redirect("https://api.example.test/", "/y", hop=3, previous=hop2)
    assert [hop1.allowed, hop2.allowed, hop3.allowed] == [True, True, False]
    assert hop3.reasons == ("DENY_REDIRECT_LIMIT",) and len(resolver.calls) == calls          # denied without a lookup
    assert [d.purpose for d in (first, hop1, hop2, hop3)] == ["navigation", "redirect", "redirect", "redirect"]
    assert resolver.calls.count("app.example.test") == 1 and resolver.calls.count("api.example.test") == 2   # fresh each hop


def test_a_redirect_to_a_listed_name_that_now_resolves_to_a_private_address_is_denied():
    resolver = E.StaticResolver({"app.example.test": [H.PUBLIC_V4], "api.example.test": ["192.168.1.1"]})
    policy = E.EgressPolicy(entries=(E.AllowEntry(schemes=("https",), ports=(443,), host="app.example.test"),
                                     E.AllowEntry(schemes=("https",), ports=(443,), host="api.example.test")))
    guard = E.EgressGuard(policy, resolver)
    first = guard.check("https://app.example.test/")
    hop = guard.check_redirect("https://app.example.test/", "https://api.example.test/", hop=1, previous=first)
    assert first.allowed and not hop.allowed and "DENY_ADDRESS_NOT_PUBLIC" in hop.reasons


def test_the_guard_refuses_a_bad_policy_or_resolver():
    with pytest.raises(Exception):
        E.EgressGuard("allow everything")
    with pytest.raises(Exception):
        E.EgressGuard(E.EgressPolicy(), object())


def test_two_guards_with_the_same_inputs_make_identical_records():
    def run():
        guard, _ = H.make_guard()
        return [guard.check(u).record_hash for u in ("https://app.example.test/", "http://127.0.0.1/", "http://metadata/",
                                                     "http://compose-app:8080/", "https://api.example.test/")]
    assert run() == run()


def test_ws_and_wss_are_judged_with_the_http_and_https_entries_of_the_same_host_and_port():
    guard = E.EgressGuard(policy_for(ports=(443,), schemes=("https",)), E.StaticResolver({"app.example.test": [H.PUBLIC_V4]}))
    assert guard.check("wss://app.example.test/socket").allowed
    assert not guard.check("ws://app.example.test:443/socket").allowed         # plain ws needs an http entry
    assert not guard.check("wss://app.example.test:8443/socket").allowed

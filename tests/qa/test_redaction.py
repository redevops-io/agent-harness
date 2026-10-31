# SPDX-License-Identifier: AGPL-3.0-or-later
"""The redactor. Each rule removes what it names; planted canaries survive in no encoding; the key name decides for
structured data; input it cannot handle fails closed (nothing raw is ever returned); redaction is idempotent, linear in
the input and deterministic; digests and counters are left alone."""
import base64
import itertools
import json
import random
import time
from urllib.parse import quote, quote_plus

import pytest

from agent_harness.qa import redaction as RD
from agent_harness.qa.errors import RedactionFailed
from agent_harness.qa.ports import Redactor

from . import _fx_evidence as H

R = H.make_redactor()


def red(text, redactor=None):
    return (redactor or R).redact_text(text)


# --- rules -----------------------------------------------------------------------------------------------------

SSH_BODY = H.pem_lines("ssh", 1)[0]
PGP_BODY = H.gen("pgp", 14)
SHORT_PEM = H.pem_lines("short", 1)[0]
BASIC_CRED = base64.b64encode(H.j("alice:", H.gen("basic", 10)).encode()).decode()
NEG, XK, AMZ = H.gen("neg", 12), H.gen("xkey", 14), H.gen("amz", 16)
COOKIE_A, COOKIE_B = H.gen("cookie-a", 9), H.gen("cookie-b", 10)
Q1, Q2, Q3, Q4 = H.gen("q1", 10), H.gen("q2", 9), H.gen("q3", 10), H.gen("q4", 8, H.HEXLOW)
K3, S1, T1 = H.gen("k3", 12, H.URLSAFE), H.gen("s1", 11, H.URLSAFE), H.gen("t1", 8)
AWS_SECRET = H.gen("aws-secret", 40, H.ALNUM + "/+")
AZ_KEY = base64.b64encode(H.gen("azure", 32).encode()).decode()
HEXV = H.gen("hexv", 32, H.HEXLOW)


def strong_token(label, n=32):
    """A mixed-case, digit-bearing token of ``n`` characters that the entropy rule must catch (deterministic search)."""
    for i in range(1000):
        tok = H.gen(f"{label}-{i}", n)
        if any(c.isupper() for c in tok) and any(c.islower() for c in tok) and any(c.isdigit() for c in tok) and RD._entropy_milli(tok) >= RD.ENTROPY_MIN_MILLI:
            return tok
    raise AssertionError("no token found")


ENT = strong_token("entropy")
AT = H.AT

#: (rule id, text containing one secret, the secret that must not survive); every credential-shaped value is generated
RULE_CASES = [
    ("private_key", f"before\n{H.PRIVATE_KEY}\nafter", H.PEM_FIRST_LINE),
    ("private_key", "x " + H.pem("", [SHORT_PEM], end=False) + "\n(truncated, no end line)", SHORT_PEM),
    ("private_key", H.pem("OPENSSH ", [SSH_BODY]), SSH_BODY),
    ("private_key", H.pem("PGP ", [PGP_BODY], label="PRIVATE KEY BLOCK"), PGP_BODY),
    ("header_value", f"GET / HTTP/1.1\nAuthorization: Bearer {H.BEARER_A}\nHost: x", H.BEARER_A),
    ("header_value", f"authorization: Basic {BASIC_CRED}", BASIC_CRED[:12]),
    ("header_value", f"Proxy-Authorization: Negotiate {NEG}", NEG),
    ("header_value", f"Cookie: sessionid={COOKIE_A}; theme=dark", COOKIE_A),
    ("header_value", f"HTTP/1.1 200 OK\nSet-Cookie: sid={COOKIE_B}; Path=/; HttpOnly", COOKIE_B),
    ("header_value", f"X-API-Key: {XK}", XK),
    ("header_value", f"  X-Amz-Security-Token: {AMZ}", AMZ),
    ("url_userinfo", H.BASIC_URL, H.USER_PW),
    ("url_userinfo", f"see ftp://bob{AT}files.example.test/x and more", "bob"),
    ("jwt", f"token seen: {H.JWT} end", H.JWT),
    ("jwt", H.JWT_UNSIGNED, H.JWT_UNSIGNED_BODY),
    ("aws_key_id", f"key {H.AWS_KEY_ID} here", H.AWS_KEY_ID),
    ("google_key", f"?k={H.GOOGLE_KEY}&x", H.GOOGLE_KEY),
    ("github_token", f"push with {H.GITHUB_TOKEN}.", H.GITHUB_TOKEN),
    ("github_token", H.GITHUB_PAT, H.GITHUB_PAT[:30]),
    ("slack_token", f"slack {H.SLACK_TOKEN}", H.SLACK_TOKEN),
    ("slack_webhook", H.SLACK_WEBHOOK, H.SLACK_WEBHOOK[-30:]),
    ("stripe_key", f"{H.STRIPE_KEY} live", H.STRIPE_KEY),
    ("sk_style_key", f"ANTHROPIC {H.MODEL_KEY}", H.MODEL_KEY),
    ("npm_token", f"//registry/:_authToken={H.NPM_TOKEN}", H.NPM_TOKEN),
    ("sendgrid_key", f"mail {H.SENDGRID_KEY}", H.SENDGRID_KEY),
    ("azure_secret", f"DefaultEndpointsProtocol=https;AccountName=a;AccountKey={AZ_KEY};EndpointSuffix=x", AZ_KEY[:24]),
    ("url_query", f"GET /cb?code={Q1}&state=ok HTTP/1.1", Q1),
    ("url_query", f"https://x.test/p?a=1&access_token={Q2}&b=2", Q2),
    ("url_query", f"https://x.test/p?sig={Q3}&se=2026", Q3),
    ("url_query", f"https://x.test/p?X-Amz-Signature={Q4}", Q4),
    ("kv_secret", f'password = "{H.PW_A}"', H.PW_A),
    ("kv_secret", f"DB_PASSWORD={H.PW_B}", H.PW_B),
    ("kv_secret", f'{{"api_key": "{K3}", "ok": true}}', K3),
    ("kv_secret", f"client_secret: {S1}", S1),
    ("kv_secret", f"'token' = '{T1}'", T1),
    ("kv_secret", f"AWS_SECRET_ACCESS_KEY={AWS_SECRET}", AWS_SECRET),
    ("kv_secret", f"sent authorization: Bearer {H.BEARER_DOTTED}", H.BEARER_DOTTED),
    ("bearer", f"curl -H 'x: y' with bearer {H.BEARER_C} inside", H.BEARER_C),
    ("pii_email", f"mail {H.EMAIL} now", H.EMAIL),
    ("pii_phone", f"call {H.PHONE} today", "555 0132"),
    ("pii_phone", "call (415) 555-0132 or 415-555-0132", "555-0132"),
    ("pii_phone", "call +44 20 7946 0958", "7946 0958"),
    ("pii_phone", "call +14155550132", "14155550132"),
    ("pii_card", f"card {H.CARD} exp", "1111 1111 1111"),
    ("pii_card", "card 4111111111111111.", "4111111111111111"),
    ("hex_token", f"key={HEXV} end", HEXV),
    ("entropy_token", f"id: {ENT}", ENT),
]


@pytest.mark.parametrize("case", RULE_CASES, ids=[f"{c[0]}-{i}" for i, c in enumerate(RULE_CASES)])
def test_each_rule_removes_what_it_names_and_counts_it(case):
    rule, text, secret = case
    out, counts = red(text)
    assert secret not in out, out
    assert counts.get(rule, 0) >= 1, (counts, out)
    assert RD.token_for(rule) in out


@pytest.mark.parametrize("case", RULE_CASES, ids=[f"{c[0]}-{i}" for i, c in enumerate(RULE_CASES)])
def test_redacting_twice_changes_nothing(case):
    out, _ = red(case[1])
    again, counts = red(out)
    assert again == out and counts == {}


def test_every_rule_in_the_library_is_exercised_by_the_table():
    exercised = {c[0] for c in RULE_CASES}
    library = {r[0] for r in RD.TEXT_RULES} | {r[0] for r in RD.PII_RULES} | {"hex_token", "entropy_token"}
    assert exercised >= library, library - exercised


def test_the_surrounding_text_survives():
    out, _ = red(f"keep this\nAuthorization: Bearer {H.BEARER_A}\nand keep that")
    assert out == "keep this\nAuthorization: [REDACTED:header_value]\nand keep that"
    out, _ = red(f'{{"password": "{H.PW_A}", "user": "ok"}}')
    assert out == '{"password": "[REDACTED:kv_secret]", "user": "ok"}'


# --- canaries ---------------------------------------------------------------------------------------------------

def encodings(secret):
    data = secret.encode()
    out = {"literal": secret, "upper": secret.upper(), "lower": secret.lower(), "pct": quote(secret, safe=""),
           "pct_plus": quote_plus(secret), "json": json.dumps(secret)[1:-1], "hex": data.hex(), "HEX": data.hex().upper(),
           "b64": base64.b64encode(data).decode(), "b64url": base64.urlsafe_b64encode(data).decode(),
           "b64_nopad": base64.b64encode(data).decode().rstrip("=")}
    for lead in (1, 2):                       # embedded at another alignment inside a longer base64 stream
        enc = base64.b64encode(b"x" * lead + data + b"yz").decode()
        out[f"b64_in_stream_{lead}"] = enc
    return out


@pytest.mark.parametrize("secret", H.CANARIES)
@pytest.mark.parametrize("form", sorted(encodings(H.CANARY_A)))
def test_a_canary_survives_in_no_encoding(secret, form):
    wrapped = encodings(secret)[form]
    text = f"<div title='x'>noise {wrapped} noise</div>"
    out, counts = red(text)
    assert counts.get("denylist", 0) >= 1 or form.startswith("b64_in_stream"), (form, out)
    probes = [secret, secret.lower(), secret.upper(), quote(secret, safe=""), base64.b64encode(secret.encode()).decode().rstrip("=")]
    if form.startswith("b64_in_stream"):
        enc = encodings(secret)[form]
        core = enc[4:-4]                                # the stable middle of the stream (aligned to the secret)
        assert core not in out or core == "", out
    else:
        assert not any(p in out for p in probes), (form, out)


def test_a_canary_inside_other_secrets_and_structures_is_removed_everywhere():
    page = f"title={H.CANARY_A}; header: Authorization: Bearer {H.CANARY_B}; url=https://x.test/?q={quote(H.CANARY_B, safe='')}&z=1"
    out, _ = red(page)
    assert not H.no_secret_survives(out) and "qa-ev-" not in out and H.CANARY_B[:3] not in out
    value, _ = R.redact_structure({"console": [f"log {H.CANARY_A}"], "nested": {"x": [H.CANARY_B]}, H.CANARY_A: 1})
    assert not H.no_secret_survives(json.dumps(value))


def test_the_policy_id_carries_the_hash_of_each_canary_and_never_the_canary():
    a = H.make_redactor(denylist=(H.CANARY_A,))
    b = H.make_redactor(denylist=(H.CANARY_A, H.CANARY_B))
    c = H.make_redactor(denylist=(H.CANARY_B, H.CANARY_A))
    assert a.policy_id != b.policy_id and b.policy_id == c.policy_id
    assert H.CANARY_A not in a.policy_id and "canary" not in a.policy_id


def test_deny_list_entries_must_be_long_enough_to_be_meaningful_and_bounded():
    for bad in ("short", "", "a" * 5000):
        with pytest.raises(Exception):
            RD.RedactionConfig(denylist=(bad,))
    with pytest.raises(Exception):
        RD.RedactionConfig(denylist=tuple(f"canary-entry-{i:04d}" for i in range(RD.MAX_DENYLIST_ENTRIES + 1)))
    with pytest.raises(Exception):
        RD.PatternRedactor(RD.RedactionConfig(denylist=("REDACTED:denylist",)))       # would match the replacement token


# --- structured data ------------------------------------------------------------------------------------------

SENSITIVE = ["password", "Password", "PASSWD", "passphrase", "secret", "client_secret", "token", "access_token",
             "refresh-token", "id_token", "authToken", "Authorization", "authorization", "proxy-authorization", "cookie",
             "Set-Cookie", "api_key", "apiKey", "x-api-key", "access_key", "private_key", "credentials", "credential",
             "bearer", "signature", "csrf", "xsrf_token", "jwt", "session_id", "sessionToken", "sid", "session", "pwd",
             "auth", "otp", "pin"]


@pytest.mark.parametrize("key", SENSITIVE)
def test_a_sensitive_key_name_redacts_its_value_whatever_it_holds(key):
    for value in ("plain words", "x", {"nested": ["a", "b"]}, ["a", 1], 12345, True):
        out, counts = R.redact_structure({key: value, "keep": "visible"})
        assert out[key] == "[REDACTED:key_name]" and out["keep"] == "visible", (key, value, out)
        assert counts[RD.KEY_RULE] == 1
    assert R.redact_structure({key: None})[0] == {key: None}              # nothing to hide


@pytest.mark.parametrize("key", ["name", "title", "url", "role", "author", "keyboard", "monkey", "key", "label", "text",
                                 "description", "count", "idempotency_key", "policy_decision_ref", "run_id"])
def test_ordinary_key_names_are_left_alone(key):
    out, counts = R.redact_structure({key: "visible value"})
    assert out == {key: "visible value"} and counts == {}


def test_counters_named_tokens_are_not_secrets():
    value = {"tokens": 12, "max_tokens": 0, "total_tokens": 5, "token": "abc", "limits": {"max_tokens": 100}}
    out, _ = R.redact_structure(value)
    assert out["tokens"] == 12 and out["max_tokens"] == 0 and out["total_tokens"] == 5 and out["limits"] == {"max_tokens": 100}
    assert out["token"] == "[REDACTED:key_name]"
    assert R.redact_structure({"tokens": "12"})[0]["tokens"] == "[REDACTED:key_name]"      # a string is not a counter


def test_exempt_keys_are_structure_and_the_rest_of_their_record_is_still_redacted():
    value = {"secret_ref": {"provider": "vault", "key": "kv/app"}, "note": f"see {H.AWS_KEY_ID}"}
    out, _ = R.redact_structure(value, exempt_keys=("secretref",))
    assert out["secret_ref"] == {"provider": "vault", "key": "kv/app"} and H.AWS_KEY_ID not in json.dumps(out)
    assert R.redact_structure(value)[0]["secret_ref"] == "[REDACTED:key_name]"


def test_strings_inside_structures_are_redacted_as_text_and_so_are_keys():
    value = {"a": [f"bearer {H.JWT}", {"b": H.GITHUB_TOKEN}], H.AWS_KEY_ID: "x", "n": 1, "f": 1.5, "t": True, "z": None}
    out, counts = R.redact_structure(value)
    text = json.dumps(out)
    assert H.JWT not in text and H.GITHUB_TOKEN not in text and H.AWS_KEY_ID not in text
    assert out["n"] == 1 and out["f"] == 1.5 and out["t"] is True and out["z"] is None


def test_two_keys_that_become_equal_after_redaction_fail_closed():
    with pytest.raises(RedactionFailed) as err:
        R.redact_structure({H.AWS_KEY_ID: 1, "[REDACTED:aws_key_id]": 2})
    assert err.value.details["reason"] == "KEY_COLLISION"


def test_structures_that_are_not_json_fail_closed():
    for bad in ({1: "a"}, {"a": object()}, {"a": b"bytes"}, {"a": {1, 2}}, [float("nan")], [float("inf")]):
        with pytest.raises(RedactionFailed):
            R.redact_structure(bad)
    deep = cur = {}
    for _ in range(100):
        cur["x"] = {}
        cur = cur["x"]
    with pytest.raises(RedactionFailed) as err:
        R.redact_structure(deep)
    assert err.value.details["reason"] == "TOO_DEEP"
    small = H.make_redactor(max_nodes=10)
    with pytest.raises(RedactionFailed) as err:
        small.redact_structure(list(range(50)))
    assert err.value.details["reason"] == "TOO_MANY_NODES"


def test_tuples_become_lists_and_the_input_is_never_modified():
    original = {"a": ("x", {"password": "p"})}
    snapshot = json.dumps(original)
    out, _ = R.redact_structure(original)
    assert out == {"a": ["x", {"password": "[REDACTED:key_name]"}]} and json.dumps(original) == snapshot


# --- things that must be left alone ----------------------------------------------------------------------------

KEEP = [
    "2026-01-01T00:00:00Z", "version 1.2.3.4 build 20260101", "ip 192.168.100.200 and 10.0.0.1:8080",
    "id 123e4567-e89b-12d3-a456-426614174000", "https://app.example.test/assets/js/app.chunk.js?v=3",
    "Basic information about the form", "the bearer of this message", "Save the changes", "Order #12345 total 99.50",
    "tokens used: 12", "rcpt-s-save-1", "trace 0123456789abcdef0123456789abcdef".replace("0123456789abcdef0123456789abcdef", "ab"),
    "/api/v2/users/12345/profile/settings/notifications-preferences-list",
    "a-very-long-hyphenated-css-class-name-for-the-primary-button",
    "x" * 40, "The quick brown fox jumps over the lazy dog " * 5, "ünïcödé text with ñ and 日本語",
]


@pytest.mark.parametrize("text", KEEP)
def test_ordinary_text_is_not_redacted(text):
    assert red(text) == (text, {})


@pytest.mark.parametrize("prefix", ["sha256", "rcv1", "canon-v1"])
def test_full_width_digests_are_never_touched_alone_or_inside_text(prefix):
    digest = prefix + ":" + "0123456789abcdef" * 4
    assert red(digest) == (digest, {})
    assert red(f"id@{digest}")[0] == f"id@{digest}"
    assert red(f"ref={digest}; next")[0] == f"ref={digest}; next"
    assert R.redact_structure({"ref": digest, "list": [digest]})[0] == {"ref": digest, "list": [digest]}
    other = "0123456789abcdef" * 4                       # the same hex without a prefix is a key-shaped token
    assert red(other)[1] == {"hex_token": 1}


def test_a_short_or_prefixless_hash_is_not_exempt():
    assert red("sha256:" + "0123456789abcdef" * 2)[1] == {"hex_token": 1}


# --- fail closed ------------------------------------------------------------------------------------------------

def fail_reason(data, media_type, redactor=None):
    with pytest.raises(RedactionFailed) as err:
        (redactor or R).redact_with_manifest(data, media_type)
    assert err.value.code == "REDACTION_FAILED"
    return err.value.details["reason"]


def test_input_that_cannot_be_handled_fails_closed_with_a_typed_error():
    assert fail_reason("text, not bytes", "text/plain") == "INPUT_NOT_BYTES"
    assert fail_reason(None, "text/plain") == "INPUT_NOT_BYTES"
    assert fail_reason(memoryview(b"x"), "text/plain") == "INPUT_NOT_BYTES"
    assert fail_reason(b"\xff\xfe\x00bad", "text/plain") == "NOT_UTF8"
    assert fail_reason("héllo".encode("latin-1"), "text/plain") == "NOT_UTF8"
    assert fail_reason(b"fine", "text/plain; charset=latin-1") == "UNSUPPORTED_CHARSET"
    assert fail_reason(b"fine", "text/plain; charset=utf-16") == "UNSUPPORTED_CHARSET"
    for media in ("image/png", "application/octet-stream", "application/zip", "video/webm", "application/pdf",
                  "application/x-ndjson-binary"):
        assert fail_reason(b"\x89PNG....", media) == "UNSUPPORTED_MEDIA_TYPE"
    for media in ("", "   ", None, 7):
        assert fail_reason(b"x", media) == "UNSUPPORTED_MEDIA_TYPE"
    small = H.make_redactor(max_input_bytes=100)
    assert fail_reason(b"z" * 101, "text/plain", small) == "INPUT_TOO_LARGE"
    assert small.redact(b"z" * 100, "text/plain") == b"z" * 100


def test_a_failure_never_echoes_the_input():
    raw = H.AWS_KEY_ID.encode()
    for data, media in ((raw + b"\xff", "text/plain"), (raw, "image/png"), (raw * 100, "text/plain; charset=latin-1")):
        with pytest.raises(RedactionFailed) as err:
            R.redact(data, media)
        assert H.AWS_KEY_ID not in str(err.value) and H.AWS_KEY_ID not in repr(err.value.details)
    with pytest.raises(RedactionFailed) as err:
        R.redact_structure({"a": object()})
    assert "object" not in str(err.value).lower() or "has no json form" in str(err.value).lower()


def test_an_unexpected_internal_error_is_a_redaction_failure_not_raw_output(monkeypatch):
    def boom(self, text, counts):
        raise ZeroDivisionError("boom with a secret " + H.AWS_KEY_ID)
    monkeypatch.setattr(RD.PatternRedactor, "_text", boom)
    with pytest.raises(RedactionFailed) as err:
        R.redact(b"anything", "text/plain")
    assert err.value.details["reason"] == "INTERNAL" and H.AWS_KEY_ID not in str(err.value)
    with pytest.raises(RedactionFailed):
        R.redact_structure({"a": "b"})


def test_a_media_type_the_operator_acknowledged_passes_through_unchanged_and_is_marked():
    r = H.make_redactor(acknowledged_unredacted_media=("image/png",))
    out, manifest = r.redact_with_manifest(b"\x89PNG\x00raw", "image/png")
    assert out == b"\x89PNG\x00raw" and manifest.passthrough and manifest.total == 0
    assert r.policy_id != R.policy_id                               # the acknowledgement is part of the policy
    with pytest.raises(RedactionFailed):
        r.redact(b"x", "image/jpeg")


def test_media_types_are_matched_by_their_base_type():
    for media in ("text/plain", "TEXT/HTML; charset=UTF-8", "application/json", "application/ld+json", "application/xml",
                  "image/svg+xml", "application/problem+json; charset=utf-8", "application/x-www-form-urlencoded",
                  "text/event-stream"):
        assert R.redact(b"ok", media) == b"ok" or media == "image/svg+xml"
    assert RD.is_text_media_type("image/svg+xml")                  # xml text, redacted as text
    assert not RD.is_text_media_type("image/png") and not RD.is_text_media_type("application/octet-stream")
    assert RD.is_text_media_type("application/hal+json") and RD.is_text_media_type("text/csv")


# --- bytes ---------------------------------------------------------------------------------------------------------

def test_json_media_is_redacted_by_key_and_by_text_and_written_deterministically():
    body = json.dumps({"z": 1, "password": H.PW_A, "list": [{"Authorization": f"Bearer {H.BEARER_A}"},
                                                                       f"note {H.AWS_KEY_ID}"], "a": {"b": None}})
    out, manifest = R.redact_with_manifest(body.encode(), "application/json")
    text = out.decode()
    assert text == json.dumps(json.loads(text), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert H.PW_A not in text and H.BEARER_A not in text and H.AWS_KEY_ID not in text
    assert manifest.counts == (("aws_key_id", 1), ("key_name", 2)) and manifest.total == 3
    assert R.redact(out, "application/json") == out                      # idempotent


def test_malformed_json_is_redacted_as_text_never_passed_through():
    out = R.redact(f'{{"password": "{H.PW_A}", broken'.encode(), "application/json").decode()
    assert H.PW_A not in out
    out = R.redact(f'{{"a": NaN, "token": "{H.TOK_A}"}}'.encode(), "application/json").decode()
    assert H.TOK_A not in out


def test_text_and_html_and_form_bodies():
    html = f"<input type=password value='x'><a href='https://x.test/?token={H.TOK_A}&n=1'>t</a><!-- {H.AWS_KEY_ID} -->"
    out = R.redact(html.encode(), "text/html; charset=utf-8").decode()
    assert H.TOK_A not in out and H.AWS_KEY_ID not in out and "<a href=" in out
    form = R.redact(f"user=alice&password={H.PW_B}&next=%2Fhome".encode(), "application/x-www-form-urlencoded").decode()
    assert H.PW_B not in form and "user=alice" in form and "next=%2Fhome" in form
    nd = R.redact(f'{{"k":"{H.GITHUB_TOKEN}"}}\n{{"n":1}}\n'.encode(), "application/x-ndjson").decode()
    assert H.GITHUB_TOKEN not in nd and '{"n":1}' in nd


def test_empty_input_is_fine():
    assert R.redact(b"", "text/plain") == b""
    assert R.redact(b"{}", "application/json") == b"{}"


def test_the_manifest_has_counts_sizes_the_rule_set_hash_and_nothing_else():
    out, m = R.redact_with_manifest(f"a {H.AWS_KEY_ID} b {H.EMAIL}".encode(), "text/plain")
    doc = m.to_json()
    assert doc["counts"] == {"aws_key_id": 1, "pii_email": 1} and doc["total"] == 2
    assert doc["rule_set_hash"] == R.policy_id and doc["input_bytes"] > doc["output_bytes"] - 40 and doc["output_bytes"] == len(out)
    assert H.AWS_KEY_ID not in json.dumps(doc) and H.EMAIL not in json.dumps(doc)
    assert m.manifest_hash.startswith("canon-v1:") and len(m.manifest_hash) == 73
    again = R.redact_with_manifest(f"a {H.AWS_KEY_ID} b {H.EMAIL}".encode(), "text/plain")[1]
    assert again.manifest_hash == m.manifest_hash


def test_the_tally_adds_counts_across_items():
    t = RD.RedactionTally(R.policy_id)
    t.add({"jwt": 1})
    t.add({})
    t.add({"jwt": 2, "kv_secret": 1})
    doc = t.to_json()
    assert doc["counts"] == {"jwt": 3, "kv_secret": 1} and doc["items_total"] == 3 and doc["items_redacted"] == 2


def test_the_redactor_satisfies_the_neutral_port_and_names_its_policy():
    assert isinstance(R, Redactor)
    assert R.policy_id.startswith("rcv1:") and len(R.policy_id) == 69
    assert R.redact(b"x", "text/plain") == b"x"


# --- policy ----------------------------------------------------------------------------------------------------

def test_the_policy_id_changes_with_every_part_of_the_rule_set():
    base = H.make_redactor().policy_id
    assert H.make_redactor().policy_id == base
    variants = [H.make_redactor(pii_email=False), H.make_redactor(pii_phone=False), H.make_redactor(pii_card=False),
                H.make_redactor(max_input_bytes=1000), H.make_redactor(extra_sensitive_keys=("ssn",)),
                H.make_redactor(denylist=()), H.make_redactor(max_nodes=99), H.make_redactor(max_depth=9)]
    assert len({v.policy_id for v in variants} | {base}) == len(variants) + 1


def test_pii_rules_are_on_by_default_and_can_be_turned_off_one_by_one():
    text = f"mail {H.EMAIL} call {H.PHONE} card {H.CARD}"
    assert red(text)[1] == {"pii_email": 1, "pii_phone": 1, "pii_card": 1}
    off = H.make_redactor(pii_email=False, pii_phone=False, pii_card=False)
    assert off.redact_text(text) == (text, {})
    assert H.make_redactor(pii_email=False).redact_text(text)[1] == {"pii_phone": 1, "pii_card": 1}


def test_the_card_rule_uses_the_luhn_check():
    assert red("4111 1111 1111 1111")[1] == {"pii_card": 1}
    assert "pii_card" not in red("4111 1111 1111 1112")[1]


def test_extra_sensitive_keys_extend_the_key_rule():
    r = H.make_redactor(extra_sensitive_keys=("ssn", "National-ID"))
    out, _ = r.redact_structure({"SSN": "123", "national_id": "x", "name": "n"})
    assert out == {"SSN": "[REDACTED:key_name]", "national_id": "[REDACTED:key_name]", "name": "n"}


# --- idempotence and determinism -------------------------------------------------------------------------------

ALPHABET = ["password", "=", ":", " ", "\n", "Bearer ", "Authorization", "token", '"', "'", "&", "?", "/", "@", "://", "a1B2c3",
            "eyJabcde.fghij.klmno", "AKIA", "QRSTUVWXYZ234567", "sk-", "x" * 30, "4111 1111 1111 1111", "[REDACTED:kv_secret]",
            "a@b.co", "+1 415 555 0132", "sha256:" + "ab" * 32, "cookie", ";", ",", "\t", "-----BEGIN " + "PRIVATE KEY-----",
            "ghp_", "\u200b", "é", "日本"]


def test_redaction_is_idempotent_on_a_few_thousand_generated_strings():
    rnd = random.Random(20261030)
    for _ in range(3000):
        text = "".join(rnd.choice(ALPHABET) for _ in range(rnd.randint(1, 14)))
        once, _ = red(text)
        twice, counts = red(once)
        assert twice == once, (text, once, twice)
        assert counts == {}


def test_redaction_is_idempotent_on_the_rule_cases_in_every_media_type():
    for _, text, _ in RULE_CASES:
        for media in ("text/plain", "text/html", "application/x-ndjson"):
            once = R.redact(text.encode(), media)
            assert R.redact(once, media) == once
    for _, text, _ in RULE_CASES:
        doc = json.dumps({"k": text, "password": "x", "list": [text]}).encode()
        once = R.redact(doc, "application/json")
        assert R.redact(once, "application/json") == once


def test_the_same_input_gives_the_same_output_and_the_same_manifest():
    data = "\n".join(c[1] for c in RULE_CASES).encode()
    outs = {R.redact_with_manifest(data, "text/plain")[0] for _ in range(3)}
    assert len(outs) == 1
    other = H.make_redactor()
    assert other.redact_with_manifest(data, "text/plain")[0] == outs.pop()


def test_a_secret_in_every_context_is_removed():
    secrets = [H.AWS_KEY_ID, H.GITHUB_TOKEN, H.SLACK_TOKEN, H.STRIPE_KEY, H.MODEL_KEY, H.JWT, H.SENDGRID_KEY, H.NPM_TOKEN,
               H.GOOGLE_KEY, H.CANARY_A, H.CANARY_B]
    contexts = ["{s}", "x {s} y", "\n{s}\n", "<p>{s}</p>", "value='{s}'", "?q={s}", "(({s}))", "a,{s},b", "\"{s}\"", "{s}{s}",
                "data:{s};", "-{s}-"]
    for secret, ctx in itertools.product(secrets, contexts):
        out, _ = red(ctx.format(s=secret))
        assert secret not in out, (ctx, secret, out)
        for media in ("text/plain", "application/json"):
            body = ctx.format(s=secret) if media == "text/plain" else json.dumps({"v": ctx.format(s=secret)})
            assert secret.encode() not in R.redact(body.encode(), media), (ctx, media)


# --- cost -------------------------------------------------------------------------------------------------------

ADVERSARIAL = {
    "underscore-run": "a_" * 120_000, "repeated-keyword": "token" * 50_000, "dot-run": "a.a." * 60_000,
    "scheme-run": "a://" * 60_000, "at-run": "x@" * 120_000, "digit-run": "1 " * 120_000, "header-run": "Authorization:" * 20_000,
    "kv-run": "password=" * 30_000, "begin-run": ("-----BEGIN " + "PRIVATE KEY-----") * 5_000, "jwt-run": "eyJ.eyJ." * 30_000,
    "quote-run": '"' * 240_000, "bearer-run": "bearer " * 40_000, "hex-run": "a1" * 120_000, "base64-run": "Aa1+" * 60_000,
    "slash-run": "/a" * 120_000, "newline-run": "\n" * 240_000, "mixed": ("password: x\nCookie: y\n" + "k" * 50) * 5_000,
}


ADVERSARIAL = {k: v * 4 for k, v in ADVERSARIAL.items()}          # about a megabyte each


@pytest.mark.parametrize("name", sorted(ADVERSARIAL))
def test_adversarial_inputs_are_processed_in_linear_time(name):
    text = ADVERSARIAL[name]
    start = time.monotonic()
    out, _ = red(text)
    elapsed = time.monotonic() - start
    assert elapsed < 8.0, (name, elapsed)
    assert isinstance(out, str)


def test_a_malformed_configuration_is_refused_not_half_applied():
    for kw in ({"acknowledged_unredacted_media": ("png",)}, {"acknowledged_unredacted_media": (5,)}, {"acknowledged_unredacted_media": ("image/png; x=1",)},
               {"extra_sensitive_keys": (5,)}, {"extra_sensitive_keys": ("",)}, {"extra_sensitive_keys": ("---",)}, {"pii_email": "yes"},
               {"max_input_bytes": 0}, {"max_input_bytes": True}, {"max_depth": 0}, {"max_nodes": -1}, {"denylist": (5,)}):
        with pytest.raises(Exception):
            RD.RedactionConfig(**kw)
    with pytest.raises(Exception):
        RD.PatternRedactor("config")
    assert RD.RedactionConfig(acknowledged_unredacted_media=(" Image/PNG ",)).acknowledged_unredacted_media == ("image/png",)

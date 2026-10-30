# SPDX-License-Identifier: AGPL-3.0-or-later
"""Test infrastructure for H5 (egress policy, redaction, evidence integrity). Test code only: nothing here ships.

Secret-shaped test values are assembled from parts at run time (``j``), so no file in the patch contains a string that
a secret scanner would take for a real credential, and none carries the benchmark canary prefix. DNS is faked: the
resolvers here are tables, and the suite's network guard refuses any real lookup."""
import copy

from agent_harness.qa import blobs as B, egress as E, redaction as RD, records as R
from agent_harness.qa.vault import ArtifactVault, EvidenceGuard

from ._h2 import APP, make_config, make_env, make_spec, nav_step, save_step


def j(*parts):
    return "".join(parts)


# --- secret-shaped values (assembled at run time) -----------------------------------------------------------------
AWS_KEY_ID = j("AKI", "AQRSTUVWXYZ234567")
GOOGLE_KEY = j("AIz", "aSyA1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q")
GITHUB_TOKEN = j("gh", "p_", "aB3dE5gH7jK9mN1pQ3sT5vW7yZ9bC2eF4hJ6")
SLACK_TOKEN = j("xo", "xb-", "123456789012-abcdEFGH5678")
STRIPE_KEY = j("sk", "_li", "ve_", "A1b2C3d4E5f6G7h8I9j0")
MODEL_KEY = j("sk", "-an", "t-api03-", "Qw3Er5Ty7Ui9Op1As3Df5Gh7Jk9Lz1Xc3Vb5")
SENDGRID_KEY = j("S", "G.", "aB3dE5gH7jK9mN1pQ3sT", ".", "xY9zA1bC3dE5fG7hI9jK")
NPM_TOKEN = j("npm", "_", "aB3dE5gH7jK9mN1pQ3sT5vW7yZ9bC2eF4hJ6")
JWT = j("ey", "JhbGciOiJIUzI1NiJ9", ".", "eyJzdWIiOiJ0ZXN0LXVzZXIifQ", ".", "abcDEF123_-xyzABC456")
PRIVATE_KEY = j("-----BEGIN ", "RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEAqwerty1234567890\nasdfghjkl0987654321\n-----END ",
                "RSA PRIVATE KEY-----")
BASIC_URL = j("https://", "alice", ":", "s3cr3tPassw0rd", "@", "db.example.test/path")
CANARY_A = j("qa-h5-", "canary-", "7f3a9c2e1b8d")
CANARY_B = j("Zq9", "-secret-", "value/with+odd=chars")
CANARIES = (CANARY_A, CANARY_B)
EMAIL = j("jane.doe", "@", "example.org")
PHONE = j("+1 415 ", "555 0132")
CARD = j("4111 ", "1111 ", "1111 ", "1111")

# --- the egress policy used by the vector tables -----------------------------------------------------------------
PUBLIC_V4 = "93.184.216.34"
PUBLIC_V4_B = "93.184.216.35"
PUBLIC_V6 = "2606:2800:220:1:248:1893:25c8:1946"

RESOLVER_TABLE = {
    "app.example.test": [PUBLIC_V4],
    "api.example.test": [PUBLIC_V4_B, PUBLIC_V6],
    "evil.example.test": ["10.0.0.5"],
    "mixed.example.test": [PUBLIC_V4, "10.0.0.5"],
    "mixed6.example.test": [PUBLIC_V4, "::1"],
    "meta.example.test": ["169.254.169.254"],
    "meta6.example.test": ["fd00:ec2::254"],
    "metamix.example.test": [PUBLIC_V4, "169.254.169.254"],
    "linklocal6.example.test": ["fe80::1"],
    "mapped.example.test": ["::ffff:93.184.216.34"],
    "mappedpriv.example.test": ["::ffff:10.0.0.5"],
    "mappedmeta.example.test": ["::ffff:169.254.169.254"],
    "nat64.example.test": ["64:ff9b::5db8:d822"],
    "sixtofour.example.test": ["2002:5db8:d822::1"],
    "teredo.example.test": ["2001:0:4136:e378:8000:63bf:3fff:fdd2"],
    "loop.example.test": ["127.0.0.1"],
    "loop6.example.test": ["::1"],
    "unspec.example.test": ["0.0.0.0"],
    "cgnat.example.test": ["100.64.1.1"],
    "alibaba.example.test": ["100.100.100.200"],
    "azure.example.test": ["168.63.129.16"],
    "ula.example.test": ["fd12:3456:789a::1"],
    "mcast.example.test": ["224.0.0.1"],
    "bcast.example.test": ["255.255.255.255"],
    "docs.example.test": ["203.0.113.9"],
    "empty.example.test": [],
    "junk.example.test": ["not-an-ip"],
    "numeric.example.test": ["0x7f.1"],
    "zone.example.test": ["fe80::1%eth0"],
    "spaced.example.test": [" 93.184.216.34"],
    "str.example.test": "93.184.216.34",
    "many.example.test": ["93.184.%d.1" % i for i in range(1, 40)],
    "a.wild.example.test": [PUBLIC_V4],
    "b.a.wild.example.test": [PUBLIC_V4],
    "wild.example.test": [PUBLIC_V4],
    "evilwild.wild.example.test": ["192.168.1.1"],
    "xn--bcher-kva.example.test": [PUBLIC_V4],
    "compose-app": ["10.20.0.5"],
    "compose-app-moved": ["10.20.0.6"],
    "compose-pub": [PUBLIC_V4],
    "localhost": ["127.0.0.1"],
    "localhost6": ["::1"],
    "other-compose": ["10.20.0.5"],
}

PUBLIC_HOSTS = [n for n in RESOLVER_TABLE if "." in n and n not in ("wild.example.test",) and not n.endswith("wild.example.test")]


def make_policy(**kwargs):
    entries = []
    for host in PUBLIC_HOSTS:
        entries.append(E.AllowEntry(schemes=("http", "https"), ports=(80, 443), host=host))
    entries += [E.AllowEntry(schemes=("http", "https"), ports=(80, 443), host="*.wild.example.test"),
                E.AllowEntry(schemes=("https",), ports=(8443,), host="app.example.test"),
                E.AllowEntry(schemes=("http", "https"), ports=(80, 443), host=PUBLIC_V4),
                E.AllowEntry(schemes=("http",), ports=(80,), host="127.0.0.1"),
                E.AllowEntry(schemes=("http",), ports=(80,), host="10.0.0.1"),
                E.AllowEntry(schemes=("http",), ports=(80,), host="0.0.0.0"),
                E.AllowEntry(schemes=("http",), ports=(80,), host="169.254.169.254"),
                E.AllowEntry(schemes=("http",), ports=(80,), host="[::1]"),
                E.AllowEntry(schemes=("http",), ports=(80,), host="[::ffff:7f00:1]"),
                E.AllowEntry(schemes=("http",), ports=(80,), host="[fd00:ec2::254]"),
                E.AllowEntry(schemes=("http",), ports=(80,), host="[64:ff9b::a9fe:a9fe]"),
                E.AllowEntry(schemes=("http",), ports=(80,), host="[2002:a9fe:a9fe::1]"),
                E.AllowEntry(schemes=("http",), ports=(80,), host="[2001:0:1:2:3:4:5601:5601]"),
                E.AllowEntry(schemes=("http",), ports=(80,), host="[fe80::1]"),
                # the operator's explicit private targets
                E.AllowEntry(schemes=("http",), ports=(8080,), host="compose-app", networks=("10.20.0.5",),
                             private_ok=True, label="compose-app"),
                E.AllowEntry(schemes=("http",), ports=(3000,), host="localhost", networks=("127.0.0.1",),
                             private_ok=True, label="local-dev"),
                E.AllowEntry(schemes=("http", "https"), ports=(8081,), networks=("10.30.0.0/24",), private_ok=True,
                             label="lab-net"),
                E.AllowEntry(schemes=("http",), ports=(8082,), host="169.254.169.254", networks=("169.254.169.254",),
                             private_ok=True, label="oops-metadata")]
    kwargs.setdefault("entries", tuple(entries))
    return E.EgressPolicy(**kwargs)


def make_guard(policy=None, table=None):
    resolver = E.StaticResolver(RESOLVER_TABLE if table is None else table)
    return E.EgressGuard(policy if policy is not None else make_policy(), resolver), resolver


# --- redaction ----------------------------------------------------------------------------------------------------
def make_redactor(**kw):
    kw.setdefault("denylist", CANARIES)
    return RD.PatternRedactor(RD.RedactionConfig(**kw))


def json_bytes(value):
    import json
    return json.dumps(value).encode("utf-8")


def no_secret_survives(text, secrets=CANARIES + (AWS_KEY_ID, GITHUB_TOKEN, SLACK_TOKEN, STRIPE_KEY, MODEL_KEY, JWT,
                                                  "s3cr3tPassw0rd", "MIIEowIBAAKCAQEAqwerty1234567890")):
    return [s for s in secrets if s in text]


# --- a run with the H5 wiring ---------------------------------------------------------------------------------------
def h5_env(*, guard=None, egress=None, spec=None, redactor=None, bounds=None, adapter_setup=None, table=None,
           egress_policy=None, **runner_kw):
    """An H2 environment with the evidence guard and the egress guard wired. The config carries the guards' policy ids,
    as the runner requires."""
    redactor = redactor if redactor is not None else make_redactor()
    guard = guard if guard is not None else EvidenceGuard(redactor, bounds)
    if egress is None:
        egress, _ = make_guard(egress_policy if egress_policy is not None else app_policy(), table if table is not None else APP_TABLE)
    config = make_config(egress_policy_id=egress.policy_id, redaction_policy_id=guard.policy_id)
    env = make_env(spec, config=config, adapter_setup=adapter_setup, evidence_guard=guard, egress=egress, **runner_kw)
    env.guard, env.egress, env.redactor = guard, egress, redactor
    return env


APP_TABLE = {"app.example.test": [PUBLIC_V4]}


def app_policy():
    return E.EgressPolicy(entries=(E.AllowEntry(schemes=("https",), ports=(443,), host="app.example.test"),))


def kinds(result):
    return [e["kind"] for e in result.events]


def events_of(result, kind):
    return [e for e in result.events if e["kind"] == kind]


def stream_text(result):
    import json
    return json.dumps([e["payload"] for e in result.events], sort_keys=True, ensure_ascii=False)

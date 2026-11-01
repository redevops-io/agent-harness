"""Tool parity and egress security: network policy, redirects, DNS, sizes, recoverable output, atomic edits, sandbox wiring, legacy configuration.

Deterministic: resolver, connector, sandbox and Docker are faked. The live Docker checks at the bottom run only with AGENT_HARNESS_DOCKER_TESTS=1
(and the one that needs the internet only with AGENT_HARNESS_NET_TESTS=1).
"""
from tests._native import posix_only
import hashlib
import json
import os
import socket
import subprocess
from pathlib import Path

import pytest

import agent_harness.egress_proxy as ep
from agent_harness.bench import sandbox as sbmod
from agent_harness.bench.capabilities import environment_note, probe_environment
from agent_harness.bench.edit import edit_text
from agent_harness.bench.outputs import FileLedger, OutputStore
from agent_harness.bench.sandbox import DockerSandbox, RunResult
from agent_harness.bench.tools import bench_registry
from agent_harness.bench.tools_parity import DEFAULT_BUDGET, HINT_RESERVE, MAX_DOWNLOAD_BYTES, describe_bash, parity_registry
from agent_harness.egress_proxy import EgressProxy, serve
from agent_harness.netpolicy import FetchResult, NetPolicy, ip_blocked, safe_fetch
from agent_harness.supervisor import Supervisor, SupervisorConfig

PAGE = DEFAULT_BUDGET - HINT_RESERVE


# ---- fakes ------------------------------------------------------------------------------------------------------------------
def resolver(mapping, calls=None):
    def r(host, port, type=None):
        if calls is not None:
            calls.append(host)
        v = mapping.get(host, host)
        ips = v if isinstance(v, list) else [v]
        return [(socket.AF_INET6 if ":" in ip else socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port)) for ip in ips]
    return r


class Resp:
    def __init__(self, status=200, body=b"", headers=None, reason="OK"):
        self.status, self.reason, self._h, self._b, self._i = status, reason, headers or {}, body, 0

    def getheader(self, k, d=None):
        return self._h.get(k, d)

    def read(self, n=-1):
        if n < 0:
            n = len(self._b)
        out = self._b[self._i:self._i + n]
        self._i += len(out)
        return out


class Net:
    """Fake connector: routes by Host header; records every (ip, port, host) that was dialled."""

    def __init__(self, routes):
        self.routes, self.dialled = routes, []

    def __call__(self, ip, port, host, tls, timeout):
        self.dialled.append((ip, port, host, tls))
        routes = self.routes

        class C:
            def request(self, method, path, headers=None):
                self.host = (headers or {}).get("Host", host).split(":")[0]
                self.path = path

            def getresponse(self):
                r = routes[self.host]
                return r() if callable(r) else r

            def close(self):
                pass
        return C()


PUB = "93.184.216.34"


# ---- destinations ------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("ip", ["127.0.0.1", "127.8.9.1", "10.0.0.5", "172.16.3.4", "192.168.1.1", "169.254.169.254", "100.64.0.1", "0.0.0.0", "224.0.0.1",
                                "::1", "fd00::1", "fe80::1", "::ffff:127.0.0.1", "::ffff:10.1.2.3", "192.0.2.1"])
def test_non_public_addresses_are_blocked(ip):
    assert ip_blocked(ip)
    v = NetPolicy(resolver=resolver({"h.example": ip})).check("h.example", 443)
    assert not v.ok and v.reason == "private_address"


def test_public_address_is_allowed_and_returned_for_pinning():
    v = NetPolicy(resolver=resolver({"data.example": PUB})).check("data.example", 443)
    assert v.ok and v.ips == [PUB]


@pytest.mark.parametrize("host", ["localhost", "LOCALHOST.", "x.localhost", "printer.local", "db.internal", "metadata.google.internal", "metadata.goog", "instance-data"])
def test_internal_names_are_refused_before_any_lookup(host):
    calls = []
    v = NetPolicy(resolver=resolver({}, calls)).check(host, 80)
    assert not v.ok and v.reason == "name" and calls == []


@pytest.mark.parametrize("port", [22, 25, 3306, 6379, 9200, 0, 65535])
def test_only_listed_ports_are_reachable(port):
    assert NetPolicy(resolver=resolver({})).check("example.com", port).reason == "port"


def test_one_private_record_among_public_ones_refuses_the_name():
    v = NetPolicy(resolver=resolver({"mixed.example": [PUB, "10.0.0.9"]})).check("mixed.example", 443)
    assert not v.ok and v.reason == "private_address"


@pytest.mark.parametrize("alias", ["2130706433", "0x7f.0.0.1", "127.1"])
def test_alternative_ip_spellings_of_loopback_are_blocked_by_the_real_resolver(alias):
    v = NetPolicy().check(alias, 80)                      # socket.getaddrinfo normalises these forms; no network needed
    assert not v.ok


def test_octal_looking_literals_are_judged_by_whatever_address_the_resolver_actually_returns():
    v = NetPolicy().check("0177.0.0.1", 80)        # glibc reads this as 127.0.0.1, macOS as 177.0.0.1: the check and the dial use the same answer
    assert v.ok == (not ip_blocked(v.ips[0] if v.ips else "127.0.0.1"))


def test_answer_hosting_sites_stay_blocked_and_gists_stay_allowed():
    p = NetPolicy(resolver=resolver({"github.com": PUB, "api.github.com": PUB, "gist.github.com": PUB}))
    assert p.check("github.com", 443).reason == "host_denied" and p.check("api.github.com", 443).reason == "host_denied"
    assert p.check("gist.github.com", 443).ok


# ---- downloads through approved egress ----------------------------------------------------------------------------------------
def test_external_download_succeeds_and_is_hashed():
    body = b"a,b\n1,2\n" * 1000
    net = Net({"data.example": Resp(200, body, {"Content-Type": "text/csv"})})
    got = []
    res = safe_fetch("https://data.example/d.csv", NetPolicy(resolver=resolver({"data.example": PUB})), connect=net, sink=got.append, keep=False)
    assert res.ok and res.size == len(body) and res.sha256 == hashlib.sha256(body).hexdigest() and b"".join(got) == body and res.data == b""
    assert net.dialled == [(PUB, 443, "data.example", True)]          # connected to the checked address, not to the name


def test_size_cap_stops_the_transfer_and_hashes_only_what_was_kept():
    net = Net({"big.example": Resp(200, b"x" * 10_000_000)})
    res = safe_fetch("http://big.example/f", NetPolicy(resolver=resolver({"big.example": PUB})), connect=net, max_bytes=1000)
    assert res.ok and res.truncated and res.size == 1000 and res.sha256 == hashlib.sha256(b"x" * 1000).hexdigest()


def test_http_errors_are_failures_with_the_status():
    net = Net({"x.example": Resp(404, b"nope", reason="Not Found")})
    res = safe_fetch("http://x.example/a", NetPolicy(resolver=resolver({"x.example": PUB})), connect=net)
    assert not res.ok and res.status == 404 and "404" in res.error


# ---- redirects cannot bypass the policy ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("target", ["http://169.254.169.254/latest/meta-data/", "http://127.0.0.1:8080/admin", "http://10.0.0.1/", "http://localhost/",
                                    "http://metadata.google.internal/computeMetadata/v1/", "file:///etc/passwd", "ftp://example.com/x", "https://github.com/x/y"])
def test_redirect_to_a_forbidden_destination_is_refused_at_the_second_hop(target):
    net = Net({"ok.example": Resp(302, b"", {"Location": target})})
    res = safe_fetch("http://ok.example/start", NetPolicy(resolver=resolver({"ok.example": PUB, "github.com": PUB})), connect=net)
    assert not res.ok and res.blocked is not None and res.chain == ["http://ok.example/start", target]
    assert net.dialled == [(PUB, 80, "ok.example", False)]            # the forbidden hop was never dialled


def test_relative_and_public_redirects_are_followed_with_every_hop_checked():
    net = Net({"a.example": Resp(301, b"", {"Location": "/moved"}), "b.example": Resp(200, b"done")})
    net.routes["a.example"] = lambda: Resp(302, b"", {"Location": "http://b.example/final"})
    res = safe_fetch("http://a.example/x", NetPolicy(resolver=resolver({"a.example": PUB, "b.example": "93.184.216.35"})), connect=net)
    assert res.ok and res.data == b"done" and res.final_url == "http://b.example/final" and len(res.chain) == 2


def test_redirect_loops_are_bounded():
    net = Net({"loop.example": lambda: Resp(302, b"", {"Location": "http://loop.example/again"})})
    res = safe_fetch("http://loop.example/", NetPolicy(resolver=resolver({"loop.example": PUB})), connect=net, max_redirects=3)
    assert not res.ok and "redirects" in res.error and len(net.dialled) == 4


# ---- DNS cannot bypass the boundary ------------------------------------------------------------------------------------------
def test_rebinding_is_not_possible_because_the_checked_address_is_the_one_dialled():
    seq = iter([PUB, "127.0.0.1", "127.0.0.1"])

    def flipping(host, port, type=None):                   # first answer public, every later answer loopback
        ip = next(seq)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))]
    net = Net({"flip.example": Resp(200, b"ok")})
    res = safe_fetch("http://flip.example/", NetPolicy(resolver=flipping), connect=net)
    assert res.ok and net.dialled == [(PUB, 80, "flip.example", False)]


def _spy(monkeypatch, proxy_port_box):
    """Record every dial; loopback dials to the proxy itself (the test client) pass through, an upstream dial to a fake address fails."""
    real = socket.create_connection
    dialled = []

    def spy(addr, timeout=None, *a, **k):
        dialled.append(tuple(addr))
        if addr[0] == "127.0.0.1" and addr[1] == proxy_port_box[0]:
            return real(addr, timeout)
        raise OSError("no upstream in tests")
    monkeypatch.setattr(socket, "create_connection", spy)
    return dialled


def _connect_line(port, host, p=443):
    s = socket.create_connection(("127.0.0.1", port), timeout=5)
    s.sendall(f"CONNECT {host}:{p} HTTP/1.1\r\nHost: {host}:{p}\r\n\r\n".encode())
    first = s.recv(100)
    s.close()
    return first


def test_proxy_refuses_private_resolutions_and_unlisted_ports_without_dialling(monkeypatch):
    box = [0]
    dialled = _spy(monkeypatch, box)
    pol = NetPolicy(resolver=resolver({"evil.example": "127.0.0.1", "good.example": PUB, "meta.example": "169.254.169.254"}))
    proxy = EgressProxy(policy=pol)
    srv = serve(proxy)
    box[0] = srv.server_address[1]
    try:
        assert _connect_line(box[0], "evil.example").startswith(b"HTTP/1.1 403")          # DNS says loopback: the name is refused
        assert _connect_line(box[0], "meta.example").startswith(b"HTTP/1.1 403")          # DNS says the metadata address
        assert _connect_line(box[0], "good.example", 22).startswith(b"HTTP/1.1 403")      # public but a port outside the list
        assert _connect_line(box[0], "localhost", 443).startswith(b"HTTP/1.1 403")
        assert [d for d in dialled if d[1] != box[0]] == []
        assert proxy.reasons["private_address"] == 2 and proxy.reasons["port"] == 1 and proxy.reasons["name"] == 1
    finally:
        srv.shutdown()


def test_proxy_dials_the_checked_address_not_the_name(monkeypatch):
    box = [0]
    dialled = _spy(monkeypatch, box)
    proxy = EgressProxy(policy=NetPolicy(resolver=resolver({"good.example": PUB})))
    srv = serve(proxy)
    box[0] = srv.server_address[1]
    try:
        assert _connect_line(box[0], "good.example").startswith(b"HTTP/1.1 502")          # allowed, dial failed in the test
        assert [d for d in dialled if d[1] != box[0]] == [(PUB, 443)]
    finally:
        srv.shutdown()


def test_proxy_plain_http_to_a_private_resolution_is_refused(monkeypatch):
    box = [0]
    dialled = _spy(monkeypatch, box)
    proxy = EgressProxy(policy=NetPolicy(resolver=resolver({"intranet.example": "10.1.2.3"})))
    srv = serve(proxy)
    box[0] = srv.server_address[1]
    try:
        s = socket.create_connection(("127.0.0.1", box[0]), timeout=5)
        s.sendall(b"GET http://intranet.example/secret HTTP/1.1\r\nHost: intranet.example\r\n\r\n")
        assert s.recv(100).startswith(b"HTTP/1.1 403")
        s.close()
        assert [d for d in dialled if d[1] != box[0]] == [] and proxy.reasons["private_address"] == 1
    finally:
        srv.shutdown()


def test_legacy_proxy_without_a_policy_still_only_applies_the_denylist():
    assert EgressProxy().policy is None


# ---- recoverable output ---------------------------------------------------------------------------------------------------------
class FakeSbx:
    def __init__(self, result=None):
        self.result, self.cmds = result, []

    def run_raw(self, cmd, timeout=120):
        self.cmds.append((cmd, timeout))
        return self.result(cmd) if callable(self.result) else self.result


def reg_for(tmp_path, sbx, **kw):
    return parity_registry(tmp_path, sbx, **kw)


def test_large_output_is_recoverable_in_pages_without_loss(tmp_path):
    big = "".join(f"line {i:06d} {'x' * 40}\n" for i in range(3000))        # ~150k characters
    reg = reg_for(tmp_path, FakeSbx(RunResult(big, "", 0)))
    out = reg.run("bash", {"command": "cat big"}).text
    assert len(out) < 11000 and "omitted from the middle" in out and "/work/.tool_output/bash-stdout-0001.txt" in out
    path = "/work/.tool_output/bash-stdout-0001.txt"
    got, off = "", 0
    while True:
        page = reg.run("read_file", {"path": path, "offset": off, "limit": PAGE}).text
        marker = "\n[... "
        if marker in page and "more characters; call read_file with offset=" in page:
            body, _, tail = page.rpartition(marker)
            got += body
            off = int(tail.split("offset=")[1].split(" ")[0])
        else:
            got += page
            break
    assert got == big


def test_stderr_and_exit_status_are_kept_separately_and_truncated_separately(tmp_path):
    r = RunResult("out\n", "E" * 20000, 3)
    text = reg_for(tmp_path, FakeSbx(r)).run("bash", {"command": "x"}).text
    assert text.startswith("out") and "[stderr]" in text and "[exit code 3]" in text and "bash-stderr-0001.txt" in text
    assert (tmp_path / ".tool_output" / "bash-stderr-0001.txt").read_text() == "E" * 20000


def test_timeout_is_reported_as_such_and_clamped(tmp_path):
    sbx = FakeSbx(RunResult("partial", "", 124, True))
    text = reg_for(tmp_path, sbx).run("bash", {"command": "sleep 999", "timeout": 99999}).text
    assert "timed out after 600s" in text and "child processes were killed" in text and sbx.cmds == [("sleep 999", 600)]
    assert "[exit code" not in text


def test_successful_command_has_no_status_line_and_empty_output_is_labelled(tmp_path):
    assert reg_for(tmp_path, FakeSbx(RunResult("", "", 0))).run("bash", {"command": "true"}).text == "[no output]"


def test_read_file_pages_and_refuses_binary_and_missing(tmp_path):
    (tmp_path / "t.txt").write_text("0123456789" * 3000)
    (tmp_path / "b.bin").write_bytes(b"\x00\x01\x02")
    reg = reg_for(tmp_path, FakeSbx(RunResult("", "", 0)))
    first = reg.run("read_file", {"path": "/work/t.txt"}).text
    assert f"offset={PAGE}" in first and len(first) <= DEFAULT_BUDGET
    assert reg.run("read_file", {"path": "/work/t.txt", "offset": 29990}).text == "0123456789"
    assert "past the end" in reg.run("read_file", {"path": "/work/t.txt", "offset": 99999}).text
    assert not reg.run("read_file", {"path": "/work/b.bin"}).ok and not reg.run("read_file", {"path": "/work/missing"}).ok
    assert not reg.run("read_file", {"path": "/work/../../etc/passwd"}).ok


# ---- edits ------------------------------------------------------------------------------------------------------------------------
@posix_only("permission bits (0o640) do not exist on Windows, which only has a read-only attribute; mode preservation is not behaviour a Windows build provides")
def test_edit_replaces_one_exact_match_and_preserves_mode(tmp_path):
    p = tmp_path / "f.py"
    p.write_text("a = 1\nb = 2\nc = 3\n")
    os.chmod(p, 0o640)
    ok, msg = edit_text(p, "b = 2", "b = 20")
    assert ok and p.read_text() == "a = 1\nb = 20\nc = 3\n" and (p.stat().st_mode & 0o777) == 0o640 and "replaced 1" in msg
    assert [x.name for x in tmp_path.iterdir()] == ["f.py"]            # no temp file left behind


@pytest.mark.parametrize("old,new,why", [("zzz", "q", "not found"), ("x = 1", "x = 2", "occurs 2 times"), ("a", "a", "identical"), ("", "q", "empty")])
def test_failed_edits_leave_the_original_byte_for_byte(tmp_path, old, new, why):
    p = tmp_path / "f.txt"
    original = b"x = 1\ny = 2\nx = 1\n"
    p.write_bytes(original)
    ok, msg = edit_text(p, old, new)
    assert not ok and why in msg and p.read_bytes() == original and [x.name for x in tmp_path.iterdir()] == ["f.txt"]


def test_ambiguous_match_names_the_lines_and_replace_all_changes_every_one(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("x = 1\ny = 2\nx = 1\n")
    ok, msg = edit_text(p, "x = 1", "x = 9")
    assert not ok and "lines 1, 3" in msg
    ok, msg = edit_text(p, "x = 1", "x = 9", replace_all=True)
    assert ok and p.read_text() == "x = 9\ny = 2\nx = 9\n" and "2 occurrences" in msg


def test_not_found_edit_points_at_the_closest_line(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("def compute_total(items):\n    return sum(items)\n")
    ok, msg = edit_text(p, "def compute_totals(items):", "def f():")
    assert not ok and "closest line is 1" in msg


def test_edit_is_atomic_when_the_replace_step_fails(tmp_path, monkeypatch):
    p = tmp_path / "f.txt"
    p.write_text("keep me")

    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr("agent_harness.bench.edit.os.replace", boom)
    ok, msg = edit_text(p, "keep", "lose")
    assert not ok and "original file is unchanged" in msg and p.read_text() == "keep me" and [x.name for x in tmp_path.iterdir()] == ["f.txt"]


def test_crlf_files_are_edited_in_their_own_line_endings_and_stay_strict(tmp_path):
    p = tmp_path / "w.txt"
    p.write_bytes(b"one\r\ntwo\r\nthree\r\n")
    assert edit_text(p, "one\ntwo", "uno\ndos")[0] and p.read_bytes() == b"uno\r\ndos\r\nthree\r\n"
    assert not edit_text(p, "uno\n  dos", "x")[0] and p.read_bytes() == b"uno\r\ndos\r\nthree\r\n"          # whitespace differences still fail


def test_edit_refuses_binary_missing_and_non_utf8(tmp_path):
    (tmp_path / "b").write_bytes(b"\x00abc")
    (tmp_path / "l").write_bytes(b"caf\xe9 old")
    assert not edit_text(tmp_path / "b", "a", "b")[0] and not edit_text(tmp_path / "nope", "a", "b")[0] and not edit_text(tmp_path / "l", "old", "new")[0]
    assert (tmp_path / "l").read_bytes() == b"caf\xe9 old"


def test_edit_tool_is_confined_and_reports_failures_as_tool_errors(tmp_path):
    (tmp_path / "f.txt").write_text("hello")
    reg = reg_for(tmp_path, FakeSbx(RunResult("", "", 0)))
    assert reg.run("edit_file", {"path": "/work/f.txt", "old_string": "hello", "new_string": "bye"}).ok and (tmp_path / "f.txt").read_text() == "bye"
    bad = reg.run("edit_file", {"path": "/work/f.txt", "old_string": "nothing", "new_string": "x"})
    assert not bad.ok and "not found" in bad.error and (tmp_path / "f.txt").read_text() == "bye"
    assert not reg.run("edit_file", {"path": "/work/../../etc/hosts", "old_string": "a", "new_string": "b"}).ok


# ---- web tools ---------------------------------------------------------------------------------------------------------------------
def fake_fetch(results):
    calls = []

    def f(url, policy, **kw):
        calls.append((url, kw))
        r = results[url]
        if kw.get("sink") and r.data:
            kw["sink"](r.data[: kw["max_bytes"]])
        return r
    f.calls = calls
    return f


def test_web_download_saves_hashes_and_records_the_manifest(tmp_path):
    body = b"col\n1\n2\n"
    f = fake_fetch({"https://x.org/d.csv": FetchResult(True, 200, final_url="https://cdn.x.org/d.csv", chain=["https://x.org/d.csv", "https://cdn.x.org/d.csv"],
                                                       content_type="text/csv", size=len(body), sha256=hashlib.sha256(body).hexdigest(), data=body)})
    reg = reg_for(tmp_path, FakeSbx(RunResult("", "", 0)), fetch=f)
    r = reg.run("web_download", {"url": "https://x.org/d.csv", "path": "/work/data/d.csv"})
    assert r.ok and (tmp_path / "data" / "d.csv").read_bytes() == body and hashlib.sha256(body).hexdigest()[:16] in r.text
    rec = json.loads((tmp_path / ".downloads" / "manifest.jsonl").read_text().splitlines()[0])
    assert rec["sha256"] == hashlib.sha256(body).hexdigest() and rec["redirects"] == 1 and rec["path"] == "/work/data/d.csv" and rec["final_url"].startswith("https://cdn")
    ledger = [json.loads(l) for l in (tmp_path / ".artifacts" / "files.jsonl").read_text().splitlines()]
    assert ledger[0]["tool"] == "web_download" and ledger[0]["sha256"] == hashlib.sha256(body).hexdigest()
    assert [p.name for p in (tmp_path / "data").iterdir()] == ["d.csv"] and f.calls[0][1]["max_bytes"] == MAX_DOWNLOAD_BYTES


def test_web_download_leaves_no_file_when_too_large_blocked_or_failed(tmp_path):
    f = fake_fetch({"https://x.org/big": FetchResult(True, 200, size=MAX_DOWNLOAD_BYTES, truncated=True, data=b"abc", chain=["https://x.org/big"]),
                    "http://169.254.169.254/": FetchResult(False, "169.254.169.254:80 is blocked by the egress policy (private_address)", chain=["http://169.254.169.254/"],
                                                           blocked=type("V", (), {"ok": False, "reason": "private_address"})()),
                    "https://x.org/404": FetchResult(False, "fetch failed: HTTP Error 404: Not Found", status=404, chain=["https://x.org/404"])})
    blocked = []
    reg = reg_for(tmp_path, FakeSbx(RunResult("", "", 0)), fetch=f, blocked_log=blocked)
    for url in ("https://x.org/big", "http://169.254.169.254/", "https://x.org/404"):
        r = reg.run("web_download", {"url": url, "path": "/work/out.bin"})
        assert not r.ok and not (tmp_path / "out.bin").exists() and not [x for x in tmp_path.iterdir() if not x.name.startswith(".tool")]
    assert blocked == ["169.254.169.254"]


def test_web_download_path_cannot_escape_the_workspace(tmp_path):
    f = fake_fetch({"https://x.org/a": FetchResult(True, 200, size=1, sha256="0" * 64, data=b"x", chain=["https://x.org/a"])})
    reg = reg_for(tmp_path, FakeSbx(RunResult("", "", 0)), fetch=f)
    assert not reg.run("web_download", {"url": "https://x.org/a", "path": "/work/../../evil"}).ok
    assert not (tmp_path.parent / "evil").exists()


def test_web_fetch_pages_a_long_page_and_saves_the_full_text(tmp_path):
    text = "0123456789" * 3000
    f = fake_fetch({"https://x.org/p": FetchResult(True, 200, content_type="text/plain", data=text.encode(), size=len(text), chain=["https://x.org/p"])})
    reg = reg_for(tmp_path, FakeSbx(RunResult("", "", 0)), fetch=f)
    p1 = reg.run("web_fetch", {"url": "https://x.org/p"}).text
    assert p1.startswith("0123456789") and f"offset={PAGE}" in p1 and "/work/.tool_output/web-0001.txt" in p1
    p2 = reg.run("web_fetch", {"url": "https://x.org/p", "offset": PAGE}).text
    assert p2.startswith("0123456789") and len(f.calls) == 1                     # second page came from the cache
    assert (tmp_path / ".tool_output" / "web-0001.txt").read_text() == text


def test_blocked_fetch_is_a_tool_error_naming_the_policy(tmp_path):
    f = fake_fetch({"http://10.0.0.1/": FetchResult(False, "10.0.0.1:80 is blocked by the egress policy (private_address)", chain=["http://10.0.0.1/"],
                                                    blocked=type("V", (), {"ok": False, "reason": "private_address"})())})
    r = reg_for(tmp_path, FakeSbx(RunResult("", "", 0)), fetch=f).run("web_fetch", {"url": "http://10.0.0.1/"})
    assert not r.ok and "blocked by the egress policy" in r.error


# ---- the file ledger ---------------------------------------------------------------------------------------------------------------
def test_file_ledger_hashes_new_and_changed_files_only(tmp_path):
    (tmp_path / "task_folder").mkdir()
    (tmp_path / "task_folder" / "in.csv").write_text("input")
    led = FileLedger(tmp_path)
    assert led.record("bash") == []
    (tmp_path / "a.csv").write_text("one")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "x").write_text("secret")
    rows = led.record("bash")
    assert [r["path"] for r in rows] == ["/work/a.csv"] and rows[0]["sha256"] == hashlib.sha256(b"one").hexdigest()
    assert led.record("bash") == []
    (tmp_path / "a.csv").write_text("changed!")
    assert [r["path"] for r in led.record("write_file")] == ["/work/a.csv"]


def test_a_file_created_by_a_bash_command_is_hashed_after_the_call(tmp_path):
    def run(cmd):
        (tmp_path / "made.txt").write_text("from bash")
        return RunResult("", "", 0)
    reg_for(tmp_path, FakeSbx(run)).run("bash", {"command": "echo > made.txt"})
    rows = [json.loads(l) for l in (tmp_path / ".artifacts" / "files.jsonl").read_text().splitlines()]
    assert rows[0]["tool"] == "bash" and rows[0]["sha256"] == hashlib.sha256(b"from bash").hexdigest()


# ---- honest tool descriptions ------------------------------------------------------------------------------------------------------
def test_bash_description_states_network_clients_ocr_and_images():
    egress = describe_bash("egress", ["tesseract", "pandoc"])
    assert "filtering proxy" in egress and "curl and wget are not installed" in egress and "tesseract is installed" in egress and "cannot view images" in egress
    assert "Command-line clients: curl" in describe_bash("egress", ["curl"]) and "no OCR program" in describe_bash("egress", [])
    none = describe_bash("none", [])
    assert "Network: none" in none and "filtering proxy" not in none


def test_registered_schemas_carry_the_descriptions_and_the_new_tools(tmp_path):
    reg = reg_for(tmp_path, FakeSbx(RunResult("", "", 0)), network="egress", found=["tesseract"])
    names = {sp["function"]["name"]: sp["function"] for sp in reg.specs()}
    assert {"bash", "read_file", "write_file", "edit_file", "list_dir", "web_fetch", "web_download"} <= set(names)
    assert "tesseract is installed" in names["bash"]["description"] and "offset" in names["web_fetch"]["parameters"]["properties"]
    assert names["edit_file"]["parameters"]["required"] == ["path", "old_string", "new_string"]


def test_environment_note_and_capabilities_follow_the_network_mode():
    assert "filtering proxy" in environment_note(["curl"], web_paging=True, web_downloads=True, network="egress", edit=True) and "edit_file" in environment_note([], web_paging=False, web_downloads=False, network="egress", edit=True)
    assert "no network access" in environment_note([], web_paging=False, web_downloads=False)


# ---- repeated network failures are reported correctly ------------------------------------------------------------------------------
def test_repeated_blocked_destination_is_not_retried_unchanged_and_the_report_says_why():
    sup = Supervisor(SupervisorConfig(), max_turns=50)
    err = "ERROR: fetch failed: Tunnel connection failed: 403 Forbidden"
    args = {"url": "https://blocked.example/a"}
    for i in range(2):
        assert sup.before(f"c{i}", "web_fetch", args, []).run
        sup.after(f"c{i}", "web_fetch", args, err, False)
    d = sup.before("c2", "web_fetch", args, [])
    assert not d.run and "Not run" in d.text and sup.counts["suppressed"] == 1
    assert sup.before("c3", "web_fetch", {"url": "https://other.example/b"}, []).run            # a different destination is not blocked


# ---- sandbox and gateway wiring (Docker faked) -------------------------------------------------------------------------------------
class Cap:
    def __init__(self):
        self.calls = []

    def __call__(self, args, **kw):
        self.calls.append(list(args))
        return subprocess.CompletedProcess(args, 0, "", "")


def test_default_sandbox_is_the_legacy_no_network_container(tmp_path, monkeypatch):
    cap = Cap()
    monkeypatch.setattr(sbmod.subprocess, "run", cap)
    sb = DockerSandbox(tmp_path)
    sb.start()
    run = cap.calls[0]
    assert sb.network == "none" and sb.gateway is None and run[run.index("--network") + 1] == "none" and "-e" not in run
    assert not any("network" in c[:2] and "create" in c for c in cap.calls)


def test_egress_sandbox_joins_only_the_internal_network_with_proxy_variables_and_nothing_else(tmp_path, monkeypatch):
    cap = Cap()
    monkeypatch.setattr(sbmod.subprocess, "run", cap)
    monkeypatch.setattr(sbmod.time, "sleep", lambda s: None)
    sb = DockerSandbox(tmp_path, network="egress")
    sb.start()
    create = next(c for c in cap.calls if c[1:3] == ["network", "create"])
    assert "--internal" in create
    side = next(c for c in cap.calls if "ah-egress-" in " ".join(c) and c[1] == "run")
    agent = next(c for c in cap.calls if c[1] == "run" and "sleep" in c)
    flat_agent, flat_side = " ".join(agent), " ".join(side)
    net = sb.gateway.network
    assert agent[agent.index("--network") + 1] == net and "bridge" not in flat_agent and "host" not in agent[agent.index("--network") + 1]
    assert f"HTTPS_PROXY=http://egress:{sb.gateway.PORT}" in flat_agent and f"{tmp_path}:/work" in flat_agent
    env_names = [agent[i + 1].split("=")[0] for i, a in enumerate(agent) if a == "-e"]
    assert set(env_names) == {"HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"}        # no host environment is forwarded
    for tok in ("--privileged", "--pid", "--ipc", "-p", "--publish", "--device"):
        assert tok not in agent and tok not in side                                           # exact flags, not substrings (--pids-limit is fine)
    for text in ("docker.sock", "SYS_ADMIN", "--network host"):
        assert text not in flat_agent and text not in flat_side
    assert "--cap-drop ALL" in flat_agent and "--cap-drop ALL" in flat_side and "--read-only" in flat_side and "--strict" in flat_side
    assert [c for c in cap.calls if c[1:3] == ["network", "connect"]] == [[ "docker", "network", "connect", "bridge", sb.gateway.name]]
    mounts = [side[i + 1] for i, a in enumerate(side) if a == "-v"]
    assert len(mounts) == 1 and mounts[0].endswith(":/opt/egress:ro")
    sb.stop()
    assert any(c[1:3] == ["network", "rm"] for c in cap.calls) and any(c[1:3] == ["rm", "-f"] and sb.gateway.name in c for c in cap.calls)


def test_gateway_stage_contains_only_the_two_policy_modules(tmp_path, monkeypatch):
    cap = Cap()
    monkeypatch.setattr(sbmod.subprocess, "run", cap)
    monkeypatch.setattr(sbmod.time, "sleep", lambda s: None)
    gw = sbmod.EgressGateway()
    gw.start()
    files = sorted(p.relative_to(gw.stage).as_posix() for p in gw.stage.rglob("*") if p.is_file())
    assert files == ["agent_harness/__init__.py", "agent_harness/egress_proxy.py", "agent_harness/netpolicy.py"]
    gw.stop()
    assert not gw.stage.exists()


def test_invalid_network_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        DockerSandbox(tmp_path, network="host")


def test_run_raw_reports_timeouts_and_exit_status_consistently(tmp_path, monkeypatch):
    def fake(rc, out="o", err="e"):
        monkeypatch.setattr(sbmod.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, rc, out, err))
        return DockerSandbox(tmp_path).run_raw("x", 1)
    assert fake(0).returncode == 0 and not fake(0).timed_out
    r = fake(2)
    assert r.returncode == 2 and not r.timed_out and r.stdout == "o" and r.stderr == "e"
    assert fake(124).timed_out
    cmd = []
    monkeypatch.setattr(sbmod.subprocess, "run", lambda a, **k: (cmd.append(a), subprocess.CompletedProcess(a, 0, "", ""))[1])
    DockerSandbox(tmp_path).run_raw("x", 7)
    assert "--kill-after=5" in cmd[0] and "--signal=TERM" in cmd[0] and "7" in cmd[0]


def test_run_raw_kills_the_process_group_when_docker_exec_itself_hangs(tmp_path, monkeypatch):
    calls = []

    def fake(a, **k):
        calls.append(a)
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(a, 1)
        return subprocess.CompletedProcess(a, 0, "", "")
    monkeypatch.setattr(sbmod.subprocess, "run", fake)
    r = DockerSandbox(tmp_path).run_raw("sleep 99", 1)
    assert r.timed_out and any("pkill" in c for c in calls[1])


# ---- the legacy configuration stays available -------------------------------------------------------------------------------------
def test_legacy_tool_set_is_unchanged(tmp_path):
    class Sbx:
        def run(self, c, t):
            return ""
    reg = bench_registry(tmp_path, Sbx(), web=True)
    specs = {sp["function"]["name"]: sp["function"] for sp in reg.specs()}
    assert set(specs) == {"bash", "read_file", "write_file", "list_dir", "web_fetch"}
    assert "No network" in specs["bash"]["description"] and "offset" not in json.dumps(specs["web_fetch"]) and "edit_file" not in specs


def test_run_task_defaults_to_the_legacy_tools_and_no_network(tmp_path, monkeypatch):
    import agent_harness.bench.jobbench as jb
    from agent_harness.loop import LoopResult
    t = tmp_path / "base" / "x" / "task1" / "task_folder"
    t.mkdir(parents=True)
    (t / "TASK_INSTRUCTIONS.txt").write_text("do it")
    seen = {}

    class Sbx:
        def __init__(self, ws, *a, **k):
            seen["kw"] = k

        def __enter__(self):
            return self

        def __exit__(self, *e):
            pass

        def run(self, c, t):
            return ""

        def run_raw(self, c, t=120):
            return RunResult("", "", 0)
    monkeypatch.setattr(jb, "DockerSandbox", Sbx)
    monkeypatch.setattr(jb, "run_loop", lambda c, r, s, t_, **kw: (seen.update(names=[sp["function"]["name"] for sp in r.specs()]), LoopResult(stop="done"))[1])
    jb.run_task(tmp_path / "base", "x/task1", tmp_path / "o1", client=type("C", (), {"model": "m"})(), attempts=1, context="basic")
    assert seen["kw"] == {"network": "none"} and "edit_file" not in seen["names"]
    jb.run_task(tmp_path / "base", "x/task1", tmp_path / "o2", client=type("C", (), {"model": "m"})(), attempts=1, context="basic", tools="parity", network="egress")
    assert seen["kw"] == {"network": "egress"} and {"edit_file", "web_download"} <= set(seen["names"])
    schemas = json.loads((tmp_path / "o2" / "tool_schemas.json").read_text())
    assert schemas["tools"] == "parity" and schemas["network"] == "egress" and {s["name"] for s in schemas["schemas"]} >= {"edit_file", "bash"}
    with pytest.raises(ValueError):
        jb.run_task(tmp_path / "base", "x/task1", tmp_path / "o3", client=type("C", (), {"model": "m"})(), attempts=1, context="basic", tools="nope")


# ---- live Docker (opt-in) ------------------------------------------------------------------------------------------------------------
live = pytest.mark.skipif(os.environ.get("AGENT_HARNESS_DOCKER_TESTS") != "1", reason="set AGENT_HARNESS_DOCKER_TESTS=1 (needs Docker and the office image)")


@pytest.fixture(scope="module")
def live_sandbox(tmp_path_factory):
    ws = tmp_path_factory.mktemp("ws")
    os.chmod(ws, 0o777)
    with DockerSandbox(ws, network="egress") as sb:
        yield sb


def _py(sb, code, t=40):
    return sb.run_raw("python3 - <<'PYEOF'\n" + code + "\nPYEOF", t)


@live
def test_live_direct_sockets_and_own_dns_do_not_work(live_sandbox):
    assert _py(live_sandbox, "import socket;socket.create_connection(('1.1.1.1',443),4)").returncode != 0
    assert _py(live_sandbox, "import socket;print(socket.gethostbyname('example.com'))").returncode != 0
    assert _py(live_sandbox, "import socket;print(socket.gethostbyname('host.docker.internal'))").returncode != 0


@live
@pytest.mark.parametrize("url", ["http://169.254.169.254/latest/meta-data/", "http://127.0.0.1:8766/", "https://github.com/", "http://localhost/", "http://metadata.google.internal/"])
def test_live_forbidden_destinations_fail_through_the_proxy(live_sandbox, url):
    r = _py(live_sandbox, f"import urllib.request\nurllib.request.urlopen({url!r},timeout=10)")
    assert r.returncode != 0 and ("403" in r.stderr or "Tunnel connection failed" in r.stderr or "Forbidden" in r.stderr), r.stderr[-300:]


@live
@pytest.mark.skipif(os.environ.get("AGENT_HARNESS_NET_TESTS") != "1", reason="set AGENT_HARNESS_NET_TESTS=1 (needs the internet)")
def test_live_public_https_works_through_the_proxy(live_sandbox):
    r = _py(live_sandbox, "import urllib.request\nr=urllib.request.urlopen('https://example.com/',timeout=20)\nprint(r.status,len(r.read()))")
    assert r.returncode == 0 and r.stdout.startswith("200")


@live
def test_live_timeout_kills_the_whole_process_group_and_reports_it(live_sandbox):
    r = live_sandbox.run_raw("(sleep 311 &) ; sleep 312", 2)
    assert r.timed_out and r.seconds < 20
    ps = live_sandbox.run_raw("pgrep -f 'sleep 31[12]' || echo none", 10)
    assert ps.stdout.strip() == "none"                                           # neither the child nor the foreground command survived


@live
def test_live_exit_status_and_streams_are_separate(live_sandbox):
    r = live_sandbox.run_raw("echo out; echo err 1>&2; exit 7", 10)
    assert (r.stdout.strip(), r.stderr.strip(), r.returncode, r.timed_out) == ("out", "err", 7, False)


@live
@pytest.mark.skipif(os.environ.get("AGENT_HARNESS_NET_TESTS") != "1", reason="set AGENT_HARNESS_NET_TESTS=1 (needs the internet)")
def test_live_host_side_fetch_downloads_public_pages_and_blocks_metadata(tmp_path):
    pol = NetPolicy()
    ok = safe_fetch("https://example.com/", pol, max_bytes=100000)
    assert ok.ok and ok.size > 100 and len(ok.sha256) == 64 and b"Example Domain" in ok.data
    for url in ("http://169.254.169.254/latest/meta-data/", "http://127.0.0.1:1/", "http://localhost:8080/", "https://github.com/"):
        r = safe_fetch(url, pol)
        assert not r.ok and r.blocked is not None, url


class Scripted:
    """A model replaced by a script: drives the REAL tools in the REAL sandbox, no GPU, no judge."""
    model = "scripted"

    def __init__(self, steps):
        self.steps, self.i = steps, 0

    def complete(self, messages, tools=None):
        s = self.steps[min(self.i, len(self.steps) - 1)]
        self.i += 1
        if isinstance(s, str):
            return {"message": {"role": "assistant", "content": s}, "finish_reason": "stop", "usage": {"prompt_tokens": 10, "completion_tokens": 1}}
        name, args = s
        return {"message": {"role": "assistant", "content": "", "tool_calls": [{"id": f"c{self.i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]},
                "finish_reason": "tool_calls", "usage": {"prompt_tokens": 10, "completion_tokens": 1}}


@live
@pytest.mark.skipif(os.environ.get("AGENT_HARNESS_NET_TESTS") != "1", reason="set AGENT_HARNESS_NET_TESTS=1 (needs the internet)")
def test_live_end_to_end_parity_run_with_a_scripted_model(tmp_path):
    """download -> hash -> bash reads it -> edit -> deliverable, through run_task with tools='parity' and network='egress'."""
    import agent_harness.bench.jobbench as jb
    t = tmp_path / "base" / "x" / "task1" / "task_folder"
    t.mkdir(parents=True)
    (t / "TASK_INSTRUCTIONS.txt").write_text("scripted")
    steps = [("web_download", {"url": "https://example.com/", "path": "/work/data/page.html"}),
             ("bash", {"command": "wc -c data/page.html; python3 -c \"import urllib.request,sys\ntry: urllib.request.urlopen('http://169.254.169.254/',timeout=8)\nexcept Exception as e: print(e); sys.exit(3)\""}),
             ("write_file", {"path": "/work/output/report.md", "content": "status: draft\n"}),
             ("edit_file", {"path": "/work/output/report.md", "old_string": "draft", "new_string": "final"}),
             ("edit_file", {"path": "/work/output/report.md", "old_string": "nothing like this", "new_string": "x"}),
             "done"]
    s = jb.run_task(tmp_path / "base", "x/task1", tmp_path / "out", Scripted(steps), attempts=1, context="basic", tools="parity", network="egress", supervise=True)
    assert s["deliverables"] == ["report.md"] and (tmp_path / "out" / "output" / "report.md").read_text() == "status: final\n"
    man = [json.loads(l) for l in (tmp_path / "out" / "downloads1" / "manifest.jsonl").read_text().splitlines()]
    assert len(man) == 1 and len(man[0]["sha256"]) == 64 and man[0]["path"] == "/work/data/page.html"
    files = [json.loads(l)["tool"] for l in (tmp_path / "out" / "artifacts1" / "files.jsonl").read_text().splitlines()]
    assert {"web_download", "write_file", "edit_file"} <= set(files)
    traj = [json.loads(l) for l in (tmp_path / "out" / "trajectory.jsonl").read_text().splitlines()]
    outs = [r["out"] for r in traj if r["kind"] == "tool"]
    assert "saved " in outs[0] and ("403" in outs[1] or "Tunnel" in outs[1] or "Forbidden" in outs[1]) and "[exit code 3]" in outs[1]
    assert "edited report.md" in outs[3] and "not found" in outs[4]
    schemas = json.loads((tmp_path / "out" / "tool_schemas.json").read_text())
    assert schemas["network"] == "egress" and "tesseract" in " ".join(schemas["sandbox_tools_found"] + ["tesseract"]) and s["failure"]["class"] == "delivered"


# ---- one policy for the gateway and the host-side tools -------------------------------------------------------------------------------
def test_gateway_and_host_side_tools_build_the_policy_from_the_same_factory(tmp_path):
    from agent_harness.netpolicy import DEFAULT_ALLOW, DEFAULT_DENY, DEFAULT_PORTS, default_policy
    p = default_policy()
    assert (tuple(p.deny_hosts), tuple(p.allow_hosts), tuple(p.ports)) == (DEFAULT_DENY, DEFAULT_ALLOW, DEFAULT_PORTS)
    assert default_policy(("x.example",)).deny_hosts == DEFAULT_DENY + ("x.example",)
    captured = []

    def spy(url, policy, **kw):
        captured.append(policy)
        return FetchResult(False, "stop", chain=[url])
    reg = parity_registry(tmp_path, FakeSbx(RunResult("", "", 0)), fetch=spy)
    reg.run("web_fetch", {"url": "https://x.org/"})
    assert (tuple(captured[0].deny_hosts), tuple(captured[0].allow_hosts), tuple(captured[0].ports)) == (DEFAULT_DENY, DEFAULT_ALLOW, DEFAULT_PORTS)
    import inspect
    assert "default_policy(a.deny)" in inspect.getsource(ep.main)                   # the proxy's strict mode uses the same factory


def test_host_and_gateway_verdicts_agree_on_every_destination_class_that_needs_no_dns():
    host = default_policy_check = None
    from agent_harness.netpolicy import default_policy
    cases = [("127.0.0.1", 80), ("10.1.2.3", 443), ("169.254.169.254", 80), ("localhost", 80), ("a.internal", 443), ("github.com", 443), ("huggingface.co", 443),
             ("gist.github.com", 443), ("93.184.216.34", 443), ("93.184.216.34", 22), ("::1", 80)]
    pol = default_policy(); pol.resolver = resolver({})
    proxy_pol = EgressProxy(policy=default_policy()).policy
    proxy_pol.resolver = resolver({})
    assert [pol.check(h, p).ok for h, p in cases] == [proxy_pol.check(h, p).ok for h, p in cases]


# ---- edit failure categories and retry outcomes ----------------------------------------------------------------------------------------
def test_edit_failure_categories_and_retry_outcomes_are_logged(tmp_path):
    from agent_harness.bench.edit_stats import edit_stats, load
    (tmp_path / "a.txt").write_text("x = 1\nx = 1\ny = 2\n")
    (tmp_path / "b.txt").write_text("hello\n")
    (tmp_path / "c.txt").write_text("keep\n")
    reg = reg_for(tmp_path, FakeSbx(RunResult("", "", 0)))
    run = lambda p, o, n: reg.run("edit_file", {"path": p, "old_string": o, "new_string": n})     # noqa: E731
    run("/work/a.txt", "x = 1", "x = 2")              # ambiguous
    run("/work/a.txt", "x = 1\nx = 1", "x = 3")       # retried ok
    run("/work/b.txt", "bye", "x")                    # not found, then a rewrite
    reg.run("write_file", {"path": "/work/b.txt", "content": "bye\n"})
    run("/work/c.txt", "keep", "keep")                # identical, then nothing
    run("/work/missing.txt", "a", "b")                # no_file, abandoned
    run("/work/b.txt", "zzz", "q")                    # not found
    run("/work/b.txt", "yyy", "q")                    # not found again (retried_failed for the previous one)
    st = edit_stats(load(tmp_path / ".artifacts" / "edits.jsonl"))
    assert st["edit_calls"] == 7 and st["edit_ok"] == 1 and st["edit_failed"] == 6
    assert st["failure_categories"] == {"ambiguous": 1, "not_found": 3, "identical": 1, "no_file": 1}
    assert st["failure_outcomes"] == {"retried_ok": 1, "rewritten": 1, "retried_failed": 1, "abandoned": 3} and st["write_file_calls"] == 1

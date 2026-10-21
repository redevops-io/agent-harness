import socket
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agent_harness.egress_proxy import EgressProxy, host_denied, serve


@pytest.mark.parametrize("host,denied", [
    ("huggingface.co", True), ("cdn-lfs.huggingface.co", True), ("hf.co", True), ("cdn.hf.co", True),
    ("github.com", True), ("api.github.com", True), ("raw.githubusercontent.com", True),
    ("gist.github.com", False), ("gist.githubusercontent.com", False),
    ("example.com", False), ("notgithub.com", False), ("evilhuggingface.co", False), ("GitHub.com.", True),
])
def test_host_policy(host, denied):
    assert host_denied(host) is denied


class Origin(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")


def test_plain_http_allowed_and_denied_hosts_are_blocked_and_counted():
    origin = ThreadingHTTPServer(("127.0.0.1", 0), Origin)
    threading.Thread(target=origin.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    proxy = EgressProxy(deny=("127.0.0.1",), allow=())      # deny the origin to prove the block path
    srv = serve(proxy)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({"http": f"http://127.0.0.1:{srv.server_address[1]}"}))
    try:
        with pytest.raises(urllib.error.HTTPError) as e:
            opener.open(f"http://127.0.0.1:{origin.server_address[1]}/x", timeout=5)
        assert e.value.code == 403 and proxy.denied == ["127.0.0.1"]
        proxy.deny = ()                                       # now allowed
        assert opener.open(f"http://127.0.0.1:{origin.server_address[1]}/x", timeout=5).read() == b"ok"
        assert proxy.allowed_count == 1
    finally:
        srv.shutdown()
        origin.shutdown()


def test_connect_to_denied_host_gets_403_without_dialing():
    proxy = EgressProxy()
    srv = serve(proxy)
    try:
        s = socket.create_connection(("127.0.0.1", srv.server_address[1]), timeout=5)
        s.sendall(b"CONNECT huggingface.co:443 HTTP/1.1\r\nHost: huggingface.co:443\r\n\r\n")
        assert s.recv(200).startswith(b"HTTP/1.1 403")
        assert proxy.denied == ["huggingface.co"]
        s.close()
    finally:
        srv.shutdown()

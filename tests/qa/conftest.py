# SPDX-License-Identifier: AGPL-3.0-or-later
"""QA test configuration (applies to tests/qa only; the existing harness tests are untouched).

* On Python < 3.11 the QA module cannot be imported (by design), so every test file except ``test_python_floor.py``
  is not collected there; that file runs on every supported interpreter and proves the guard.
* Network guard: any attempt to connect to a non-loopback address, or to resolve a non-loopback name,
  fails the test. Nothing here needs the network.
"""
import ipaddress
import socket
import sys
from pathlib import Path

import pytest

collect_ignore = []
if sys.version_info < (3, 11):
    collect_ignore = [p.name for p in Path(__file__).parent.glob("test_*.py") if p.name != "test_python_floor.py"]

_LOOPBACK_NAMES = {"localhost", "localhost.localdomain"}


def _is_loopback_host(host):
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    if host in (None, ""):
        return True
    if host in _LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def guard(address):
        if isinstance(address, tuple) and address and not _is_loopback_host(address[0]):
            raise AssertionError(f"tests/qa must not touch the network: connect to {address[0]!r}")

    def connect(self, address):
        guard(address)
        return real_connect(self, address)

    def connect_ex(self, address):
        guard(address)
        return real_connect_ex(self, address)

    def getaddrinfo(host, *args, **kwargs):
        if not _is_loopback_host(host):
            raise AssertionError(f"tests/qa must not resolve names: {host!r}")
        return real_getaddrinfo(host, *args, **kwargs)

    real_getaddrinfo = socket.getaddrinfo
    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    yield

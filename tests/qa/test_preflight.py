# SPDX-License-Identifier: AGPL-3.0-or-later
"""Plan gates G0 (environment hygiene) and G1 (no live hosts) for the QA tests.

The strict check of the real environment runs when ``QA_STRICT_ENV=1`` is set, as the integrator's scrubbed command
does; it is opt-in so that ordinary contributor and CI runs of the existing suite are not affected."""
import os
import re
import socket
import tempfile

import pytest

from .conftest import _is_loopback_host

SECRETISH = re.compile(r"TOKEN|SECRET|KEY|PASSWORD|CREDENTIAL", re.IGNORECASE)
CLOUD_PROFILE = re.compile(r"^(AWS_PROFILE|AWS_DEFAULT_PROFILE|AZURE_|GOOGLE_APPLICATION_CREDENTIALS|GH_TOKEN|GITHUB_TOKEN)", re.IGNORECASE)
TEMP_ROOTS = ("/tmp", "/var/tmp", "/private/tmp", "/var/folders", "/private/var/folders")


def env_problems(environ):
    problems = []
    for name in sorted(environ):
        if SECRETISH.search(name) or CLOUD_PROFILE.match(name):
            problems.append(f"variable {name} looks like a credential")
    home = environ.get("HOME", "")
    temp_like = [tempfile.gettempdir(), *TEMP_ROOTS]
    if not home or not any(os.path.realpath(home).startswith(os.path.realpath(root)) for root in temp_like):
        problems.append(f"HOME={home!r} is not a temporary directory")
    if "APPROVAL_SECRET" in environ:
        problems.append("APPROVAL_SECRET must be unset")
    return problems


def test_a_clean_environment_has_no_problems(tmp_path):
    assert env_problems({"HOME": str(tmp_path), "PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}) == []


@pytest.mark.parametrize("name", ["GH_TOKEN", "GITHUB_TOKEN", "OPENAI_API_KEY", "DB_PASSWORD", "APPROVAL_SECRET",
                                  "AWS_PROFILE", "AWS_SECRET_ACCESS_KEY", "SOME_CREDENTIALS", "GOOGLE_APPLICATION_CREDENTIALS"])
def test_credential_like_variables_are_flagged(tmp_path, name):
    assert env_problems({"HOME": str(tmp_path), name: "x"})


def test_a_real_home_directory_is_flagged():
    assert env_problems({"HOME": "/Users/someone"})
    assert env_problems({})


def test_the_real_environment_is_scrubbed_when_strict_mode_is_on():
    if os.environ.get("QA_STRICT_ENV") != "1":
        assert env_problems({"HOME": "/home/x"}) != []          # strict mode off: only the checker itself is exercised
        return
    assert env_problems(dict(os.environ)) == []


def test_loopback_detection():
    for host in ("127.0.0.1", "127.1.2.3", "::1", "localhost", "", None):
        assert _is_loopback_host(host), host
    for host in ("192.0.2.1", "10.0.0.1", "169.254.169.254", "example.com", "0.0.0.0", "::", "metadata.google.internal"):
        assert not _is_loopback_host(host), host


def test_the_network_guard_refuses_non_loopback_connections_without_touching_the_network():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(AssertionError, match="must not touch the network"):
            s.connect(("192.0.2.1", 9))
        with pytest.raises(AssertionError, match="must not touch the network"):
            s.connect_ex(("169.254.169.254", 80))
    finally:
        s.close()


def test_the_resolver_guard_refuses_names():
    with pytest.raises(AssertionError, match="must not resolve names"):
        socket.getaddrinfo("example.com", 80)
    with pytest.raises(AssertionError, match="must not resolve names"):
        socket.getaddrinfo("metadata.google.internal", 80)

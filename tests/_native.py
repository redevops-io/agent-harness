"""Markers for tests that cannot pass on a native Windows host yet.

``windows_todo`` is an expected failure (strict) on Windows only: the test states behaviour a Windows build has to
provide, and it stays visible as an expected failure until that behaviour is implemented. A strict xfail turns into
a failure when the behaviour starts working, which forces the marker to be removed. It never hides a Linux or macOS
result.

``posix_only`` is a skip for a test of something that does not exist on Windows (not behaviour a Windows build has
to provide); the reason must say why.
"""
import sys

import pytest

IS_WINDOWS = sys.platform == "win32"


def windows_todo(reason: str):
    return pytest.mark.xfail(IS_WINDOWS, reason="to implement (Windows): " + reason, strict=True)


def posix_only(reason: str):
    return pytest.mark.skipif(IS_WINDOWS, reason="POSIX only: " + reason)

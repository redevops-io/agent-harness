"""The environment a child process starts with: an explicit allowlist, never ``os.environ`` wholesale (standard library only).

Before this module the sandbox and the judge and benchmark runners passed the whole parent environment to every child, so
a credential exported in the launching shell (an API key, a cloud or source-control token) reached every command the agent
ran. :func:`child_env` starts from nothing and adds only:

* the allowlisted names for this host (a ``PATH``-equivalent, locale, temp directory and, on Windows, the profile and system
  variables a Windows program needs: ``SYSTEMROOT``, ``COMSPEC``, ``PATHEXT``, ``USERPROFILE``, ``APPDATA`` ...);
* with ``preset="bench"``, the few more that a benchmark runner's third-party CLI needs (``HOME``, proxy and CA variables ...);
* names the caller passes explicitly (``pass_through``: copied from the parent environment, ``extra``: given values), and
  names an operator lists in ``AGENT_HARNESS_ENV_PASS`` (comma separated).

Nothing whose name looks like a secret (``*KEY*``, ``*TOKEN*``, ``*SECRET*``, ``*PASSWORD*``, ``*CREDENTIAL*``, ``*AUTH*`` ...)
is ever added by the allowlist. Passing a named variable is an explicit decision and is logged (names only, never values),
at WARNING when an ambient variable that looks like a secret is passed. Setting ``AGENT_HARNESS_ENV_INHERIT=1`` (or
``inherit_all=True``) opts out of the policy entirely and passes the full environment, with a WARNING on every call.
"""
from __future__ import annotations

import fnmatch
import logging
import os
import re
import sys
from typing import Dict, Iterable, Mapping, Optional, Sequence

log = logging.getLogger("agent_harness.env_policy")

ENV_PASS_VAR = "AGENT_HARNESS_ENV_PASS"          # comma-separated extra names to pass to every child
ENV_INHERIT_VAR = "AGENT_HARNESS_ENV_INHERIT"    # "1": opt out of the policy (pass the whole environment)

IS_WINDOWS = sys.platform == "win32"

_COMMON = ("PATH", "LANG", "LANGUAGE", "LC_*", "TZ", "TERM")
_POSIX = ("TMPDIR",)
_WINDOWS = ("PATHEXT", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "TEMP", "TMP", "USERPROFILE", "APPDATA",
            "LOCALAPPDATA", "PROGRAMDATA", "PROGRAMFILES", "PROGRAMFILES(X86)", "PROGRAMW6432", "COMMONPROGRAMFILES",
            "COMMONPROGRAMFILES(X86)", "COMMONPROGRAMW6432", "ALLUSERSPROFILE", "PUBLIC", "HOMEDRIVE", "HOMEPATH", "OS",
            "PROCESSOR_ARCHITECTURE", "NUMBER_OF_PROCESSORS")
# What a benchmark runner's third-party command-line client (and the interpreter running the judge) typically needs.
_BENCH = ("HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME", "PYTHONPATH", "PYTHONUTF8",
          "PYTHONIOENCODING", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "no_proxy",
          "all_proxy", "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "NODE_EXTRA_CA_CERTS")

SECRET_PATTERNS = ("*KEY*", "*TOKEN*", "*SECRET*", "*PASSWORD*", "*PASSWD*", "*CREDENTIAL*", "*AUTH*", "*COOKIE*",
                   "*PRIVATE*", "*SIGNATURE*", "*PASS*", "*DSN*", "*DATABASE*", "*WEBHOOK*", "*CONNECTION*", "*CONN*STR*", "AWS_*", "AZURE_*", "GOOGLE_*", "GCP_*", "GITHUB_*", "GITLAB_*", "GH_*",
                   "NPM_*", "DOCKER_*", "VAULT_*", "OPENAI_*", "ANTHROPIC_*", "NIM_*", "JUDGE_*")


# Configuration values that share a credential-shaped prefix but are not credentials.
NOT_SECRET = frozenset({"JUDGE_MAX_COMPLETION_TOKENS", "JUDGE_REASONING_EFFORT"})

# scheme://user:password@host: a credential embedded in a URL value (a proxy, a database address)
_URL_CREDENTIALS = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://[^/\s:@]+:[^@\s/]+@")


def _norm(name: str) -> str:
    return name.upper() if IS_WINDOWS else name           # Windows names are case-insensitive, POSIX names are not


def _match(name: str, patterns: Iterable[str]) -> bool:
    """Allowlist matching follows the host: case-insensitive on Windows, case-sensitive on POSIX."""
    if IS_WINDOWS:
        n = name.upper()
        return any(fnmatch.fnmatchcase(n, p.upper()) for p in patterns)
    return any(fnmatch.fnmatchcase(name, p) for p in patterns)


def looks_secret(name: str) -> bool:
    """Credential-shaped names, on every host regardless of case (the patterns are written upper-case)."""
    n = name.upper()
    if n in NOT_SECRET:
        return False
    return any(fnmatch.fnmatchcase(n, p) for p in SECRET_PATTERNS)


def has_embedded_credentials(value: str) -> bool:
    return bool(_URL_CREDENTIALS.search(value or ""))


def allowed_names(preset: str = "tool") -> tuple:
    """The allowlist patterns for this host and preset (``"tool"``: the default; ``"bench"``: plus runner needs)."""
    names = _COMMON + (_WINDOWS if IS_WINDOWS else _POSIX)
    if preset == "bench":
        names += _BENCH
    elif preset != "tool":
        raise ValueError(f"unknown env preset: {preset!r}")
    return names


def default_child_path(environ: Optional[Mapping[str, str]] = None) -> str:
    """The ``PATH`` for a sandboxed child. POSIX: the fixed trusted directories. Windows: the system directories only
    (never the caller's ``PATH`` or the working directory)."""
    if IS_WINDOWS:
        env = os.environ if environ is None else environ
        root = next((v for k, v in env.items() if k.upper() == "SYSTEMROOT"), r"C:\Windows")
        return os.pathsep.join([os.path.join(root, "System32"), root, os.path.join(root, "System32", "Wbem")])
    return os.pathsep.join(("/usr/local/bin", "/usr/bin", "/bin"))


def _listed(value: str) -> list:
    return [n.strip() for n in value.split(",") if n.strip()]


def child_env(*, preset: str = "tool", path: Optional[str] = None, inherit_path: bool = False,
              pass_through: Sequence[str] = (), extra: Optional[Mapping[str, str]] = None,
              inherit_all: bool = False, environ: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """Build a child environment from an allowlist. ``path`` overrides ``PATH`` (default :func:`default_child_path`);
    ``inherit_path=True`` keeps the parent's ``PATH`` instead (for a benchmark runner that has to find ``node`` or its
    interpreter). ``pass_through`` names are copied from the parent when set; ``extra`` values are given explicitly."""
    if isinstance(pass_through, (str, bytes)):
        raise TypeError("pass_through must be a sequence of names, not a single string (got %r)" % (pass_through,))
    src = dict(os.environ if environ is None else environ)
    if inherit_all or src.get(ENV_INHERIT_VAR) == "1":
        out = dict(src)
        out.pop(ENV_INHERIT_VAR, None)
        if path is not None or not inherit_path:        # opting out of the credential policy does not unpin PATH
            out = {k: v for k, v in out.items() if _norm(k) != "PATH"}
            out["PATH"] = path if path is not None else default_child_path(src)
        if extra:
            out.update(extra)
        log.warning("env policy OFF (%s): the child inherits the whole parent environment (%d variables), including any "
                    "credentials in it", "inherit_all" if inherit_all else ENV_INHERIT_VAR + "=1", len(out))
        return out
    allow = allowed_names(preset)
    out: Dict[str, str] = {}
    for k, v in src.items():
        if _match(_norm(k), allow) and not looks_secret(k):
            out[k] = v
    if path is not None:
        out = {k: v for k, v in out.items() if _norm(k) != "PATH"}
        out["PATH"] = path
    elif not inherit_path:
        out = {k: v for k, v in out.items() if _norm(k) != "PATH"}
        out["PATH"] = default_child_path(src)
    wanted = list(pass_through) + _listed(src.get(ENV_PASS_VAR, ""))
    passed, risky = [], []
    for name in wanted:
        hit = next((k for k in src if _norm(k) == _norm(name)), None)
        if hit is None:
            continue
        out[hit] = src[hit]
        passed.append(hit)
        if looks_secret(hit):
            risky.append(hit)
    if extra:
        for k, v in extra.items():
            out[k] = v
        passed.extend(k for k in extra if k not in passed)
    leaky = sorted(k for k, v in out.items() if _norm(k) != "PATH" and has_embedded_credentials(v))
    if leaky:
        log.warning("env policy: the value of %s embeds credentials (scheme://user:password@host) and is being passed to the "
                    "child", ", ".join(leaky))
    if any(_norm(n) == "PATH" for n in passed):
        log.warning("env policy: PATH was passed explicitly (by name or as a value), so it is no longer pinned to the trusted "
                    "directories")
    if passed:
        log.info("env policy: passing %d named variable(s) to the child: %s", len(passed), ", ".join(sorted(set(passed))))
    if risky:
        log.warning("env policy: ambient variable(s) that look like credentials are being passed to the child by name: %s",
                    ", ".join(sorted(set(risky))))
    return out

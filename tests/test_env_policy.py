"""The child-environment policy: an allowlist, never ``os.environ`` wholesale; named opt-ins are logged; opting out warns."""
import json
import logging
import os
import sys

import pytest

from agent_harness import env_policy
from agent_harness.env_policy import child_env, looks_secret

PY = sys.executable
PLANTED = {"OPENAI_API_KEY": "sk-planted-1", "AWS_SECRET_ACCESS_KEY": "planted-2", "GITHUB_TOKEN": "ghp_planted3",
           "MY_SERVICE_PASSWORD": "planted-4", "SSH_AUTH_SOCK": "/tmp/planted-5", "NIM_API_KEY": "planted-6",
           "JUDGE_API_KEY": "planted-7", "MY_APP_SETTING": "not-a-secret-name-but-not-allowlisted"}


def posix_env(**more):
    return {"PATH": "/home/u/bin:/usr/bin", "LANG": "en_US.UTF-8", "LC_ALL": "C.UTF-8", "TMPDIR": "/tmp/x", "TZ": "UTC",
            "HOME": "/home/u", "SHELL": "/bin/zsh", **PLANTED, **more}


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.delenv(env_policy.ENV_PASS_VAR, raising=False)
    monkeypatch.delenv(env_policy.ENV_INHERIT_VAR, raising=False)
    monkeypatch.setattr(env_policy, "IS_WINDOWS", False)


def test_default_child_env_is_the_allowlist_and_none_of_the_planted_secrets():
    out = child_env(environ=posix_env())
    assert set(out) == {"PATH", "LANG", "LC_ALL", "TMPDIR", "TZ"}
    assert out["PATH"] == os.pathsep.join(("/usr/local/bin", "/usr/bin", "/bin"))      # the fixed trusted path, not the caller's
    assert "/home/u/bin" not in out["PATH"]
    assert not any(k in out for k in PLANTED) and "HOME" not in out and "SHELL" not in out


def test_the_bench_preset_adds_home_and_proxy_variables_but_still_no_credentials():
    out = child_env(preset="bench", inherit_path=True,
                    environ=posix_env(HTTPS_PROXY="http://127.0.0.1:8766", PYTHONPATH="/repo"))
    assert out["HOME"] == "/home/u" and out["HTTPS_PROXY"] == "http://127.0.0.1:8766" and out["PYTHONPATH"] == "/repo"
    assert out["PATH"] == "/home/u/bin:/usr/bin"                                  # the runner's own PATH, by request
    assert not any(k in out for k in PLANTED)


def test_unknown_preset_is_refused():
    with pytest.raises(ValueError):
        child_env(preset="everything", environ={})


def test_secret_looking_names():
    for n in ("OPENAI_API_KEY", "github_token", "DB_PASSWORD", "AWS_SESSION_TOKEN", "GOOGLE_APPLICATION_CREDENTIALS",
              "SSH_AUTH_SOCK", "ANTHROPIC_FOO", "NIM_BASE_URL", "JUDGE_API_KEY"):
        assert looks_secret(n), n
    for n in ("PATH", "LANG", "TMPDIR", "HOME", "USERPROFILE", "SYSTEMROOT", "HTTPS_PROXY", "PYTHONPATH"):
        assert not looks_secret(n), n


def test_naming_a_variable_passes_it_and_logs_the_name_not_the_value(caplog):
    caplog.set_level(logging.INFO, logger="agent_harness.env_policy")
    out = child_env(pass_through=["MY_APP_SETTING", "NOT_SET_ANYWHERE"], environ=posix_env())
    assert out["MY_APP_SETTING"] == "not-a-secret-name-but-not-allowlisted" and "NOT_SET_ANYWHERE" not in out
    text = caplog.text
    assert "MY_APP_SETTING" in text and "not-a-secret-name" not in text
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_passing_an_ambient_credential_by_name_works_and_warns(caplog):
    caplog.set_level(logging.INFO, logger="agent_harness.env_policy")
    out = child_env(pass_through=["OPENAI_API_KEY"], environ=posix_env())
    assert out["OPENAI_API_KEY"] == "sk-planted-1"
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings and "OPENAI_API_KEY" in warnings[0].getMessage() and "sk-planted" not in caplog.text
    assert "GITHUB_TOKEN" not in out                                           # only what was named


def test_the_operator_variable_lists_names_to_pass(caplog):
    caplog.set_level(logging.INFO, logger="agent_harness.env_policy")
    out = child_env(environ=posix_env(AGENT_HARNESS_ENV_PASS="MY_APP_SETTING, GITHUB_TOKEN"))
    assert out["MY_APP_SETTING"] and out["GITHUB_TOKEN"] == "ghp_planted3" and "OPENAI_API_KEY" not in out
    assert "AGENT_HARNESS_ENV_PASS" not in out
    assert any(r.levelno == logging.WARNING for r in caplog.records)           # GITHUB_TOKEN looks like a credential


def test_extra_values_given_explicitly_are_added_and_named_in_the_log(caplog):
    caplog.set_level(logging.INFO, logger="agent_harness.env_policy")
    out = child_env(extra={"JUDGE_API_KEY": "given", "JUDGE_REASONING_EFFORT": "low"}, environ=posix_env())
    assert out["JUDGE_API_KEY"] == "given" and out["JUDGE_REASONING_EFFORT"] == "low"
    assert "JUDGE_API_KEY" in caplog.text and "given" not in caplog.text.replace("given explicitly", "")


def test_opting_out_passes_everything_and_warns_every_time(caplog):
    caplog.set_level(logging.INFO, logger="agent_harness.env_policy")
    for kw in ({"inherit_all": True}, {}):
        caplog.clear()
        env = posix_env(**({} if kw else {"AGENT_HARNESS_ENV_INHERIT": "1"}))
        out = child_env(environ=env, **kw)
        assert out["OPENAI_API_KEY"] == "sk-planted-1" and out["HOME"] == "/home/u"
        assert "AGENT_HARNESS_ENV_INHERIT" not in out
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings and "policy OFF" in warnings[0].getMessage()


def test_windows_variables_are_kept_case_insensitively_and_path_is_the_system_path(monkeypatch):
    monkeypatch.setattr(env_policy, "IS_WINDOWS", True)
    env = {"Path": r"C:\Users\u\bin", "SystemRoot": r"C:\Windows", "ComSpec": r"C:\Windows\System32\cmd.exe",
           "PATHEXT": ".COM;.EXE", "USERPROFILE": r"C:\Users\u", "AppData": r"C:\Users\u\AppData\Roaming",
           "LOCALAPPDATA": r"C:\Users\u\AppData\Local", "ProgramData": r"C:\ProgramData", "TEMP": r"C:\Temp",
           "GITHUB_TOKEN": "planted", "OPENAI_API_KEY": "planted", "MY_APP_SETTING": "x"}
    out = child_env(environ=env)
    kept = {k.upper() for k in out}
    assert {"PATH", "SYSTEMROOT", "COMSPEC", "PATHEXT", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "PROGRAMDATA", "TEMP"} <= kept
    assert "GITHUB_TOKEN" not in out and "OPENAI_API_KEY" not in out and "MY_APP_SETTING" not in out
    assert sum(1 for k in out if k.upper() == "PATH") == 1                      # one PATH, not the caller's
    path = next(v for k, v in out.items() if k.upper() == "PATH")
    assert "System32" in path and r"C:\Users\u\bin" not in path


# ---------------------------------------------------------------------------------------------------------------
# End to end through the sandbox and the judge launcher (real child processes)
# ---------------------------------------------------------------------------------------------------------------

DUMP = "import json, os; print(json.dumps(dict(os.environ)))"


def run_dump(**kw):
    from agent_harness import sandbox
    out = sandbox.run_shell("", argv=[PY, "-c", DUMP], timeout=60, **kw)
    assert out["returncode"] == 0, out
    return json.loads(out["stdout"])


@pytest.fixture
def planted(monkeypatch):
    for k, v in PLANTED.items():
        monkeypatch.setenv(k, v)


def test_a_sandboxed_child_does_not_see_planted_credentials(planted):
    seen = run_dump()
    assert not any(k in seen for k in PLANTED), sorted(seen)
    assert "PATH" in seen
    if sys.platform != "win32":
        assert seen["PATH"] == "/usr/local/bin:/usr/bin:/bin"


def test_a_named_opt_in_reaches_the_sandboxed_child_and_is_logged(planted, caplog):
    caplog.set_level(logging.INFO, logger="agent_harness.env_policy")
    seen = run_dump(env_pass=("MY_APP_SETTING",))
    assert seen["MY_APP_SETTING"] and "OPENAI_API_KEY" not in seen and "GITHUB_TOKEN" not in seen
    assert "MY_APP_SETTING" in caplog.text


def test_opting_out_through_the_environment_variable_reaches_the_child_with_a_warning(planted, monkeypatch, caplog):
    monkeypatch.setenv("AGENT_HARNESS_ENV_INHERIT", "1")
    seen = run_dump()
    assert seen["OPENAI_API_KEY"] == "sk-planted-1"
    assert any(r.levelno == logging.WARNING and "policy OFF" in r.getMessage() for r in caplog.records)


def test_the_judge_launcher_gets_its_configured_values_but_not_ambient_credentials(planted):
    from agent_harness.bench.evalkit import JudgeConfig, _judge_env
    cfg = JudgeConfig(PY, "judge.py", "m", "http://x", api_key="configured", reasoning_effort="high")
    env = _judge_env(cfg)
    assert env["JUDGE_API_KEY"] == "configured" and env["JUDGE_REASONING_EFFORT"] == "high"
    assert "OPENAI_API_KEY" not in env and "GITHUB_TOKEN" not in env and "NIM_API_KEY" not in env


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX only: the trusted directories are POSIX paths")
def test_the_default_path_is_the_command_resolvers_trusted_path():
    from agent_harness.command_policy import TRUSTED_PATH
    assert env_policy.default_child_path() == os.pathsep.join(TRUSTED_PATH)


# ---------------------------------------------------------------------------------------------------------------
# Review fixes
# ---------------------------------------------------------------------------------------------------------------

TRUSTED = os.pathsep.join(("/usr/local/bin", "/usr/bin", "/bin"))


def test_opting_out_still_pins_path_to_the_trusted_directories():
    env = posix_env(PATH="/tmp/evil:/usr/bin")
    assert child_env(environ=env, inherit_all=True)["PATH"] == TRUSTED
    assert child_env(environ={**env, "AGENT_HARNESS_ENV_INHERIT": "1"})["PATH"] == TRUSTED
    assert child_env(environ=env, inherit_all=True, inherit_path=True)["PATH"] == "/tmp/evil:/usr/bin"   # explicit request
    assert child_env(environ=env, inherit_all=True, path="/opt/x")["PATH"] == "/opt/x"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX only: the trusted directories are POSIX paths")
def test_opting_out_through_the_environment_variable_keeps_the_sandbox_path_pinned(monkeypatch):
    from agent_harness import sandbox
    monkeypatch.setenv("AGENT_HARNESS_ENV_INHERIT", "1")
    monkeypatch.setenv("PATH", "/tmp/evil:" + os.environ["PATH"])
    out = sandbox.run_shell("", argv=[PY, "-c", "import os; print(os.environ['PATH'])"], timeout=30)
    assert out["stdout"].strip() == TRUSTED


def test_a_plain_string_is_not_split_into_variable_names():
    with pytest.raises(TypeError):
        child_env(pass_through="MY_APP_SETTING", environ=posix_env(A="1"))
    with pytest.raises(TypeError):
        child_env(pass_through=b"MY_APP_SETTING", environ=posix_env(A="1"))
    from agent_harness import sandbox
    out = sandbox.run_shell("", argv=[PY, "-c", "print(1)"], timeout=30, env_pass="MY_APP_SETTING")
    assert out["returncode"] != 0 and "sequence of names" in out["stderr"]


def test_the_judge_keeps_its_completion_token_setting_without_a_credential_warning(monkeypatch, caplog):
    from agent_harness.bench.evalkit import JudgeConfig, _judge_env
    monkeypatch.setenv("JUDGE_MAX_COMPLETION_TOKENS", "4096")
    monkeypatch.setenv("JUDGE_REASONING_EFFORT", "ambient-should-lose-to-config")
    caplog.set_level(logging.INFO, logger="agent_harness.env_policy")
    env = _judge_env(JudgeConfig(PY, "judge.py", "m", "http://x", reasoning_effort="high"))
    assert env["JUDGE_MAX_COMPLETION_TOKENS"] == "4096" and env["JUDGE_REASONING_EFFORT"] == "high"
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_more_credential_shaped_names_are_recognised():
    for n in ("DATABASE_URL", "SENTRY_DSN", "DSN", "PGPASS", "PGPASSWORD", "SLACK_WEBHOOK_URL", "DB_CONN_STR", "MY_PASS"):
        assert looks_secret(n), n
    for n in ("JUDGE_MAX_COMPLETION_TOKENS", "JUDGE_REASONING_EFFORT", "PATH", "HTTPS_PROXY"):
        assert not looks_secret(n), n


def test_a_proxy_value_with_embedded_credentials_warns_without_logging_the_value(caplog):
    caplog.set_level(logging.INFO, logger="agent_harness.env_policy")
    out = child_env(preset="bench", inherit_path=True,
                    environ={"PATH": "/usr/bin", "HTTPS_PROXY": "http://alice:s3cr3t@proxy.example:3128",
                             "HTTP_PROXY": "http://proxy.example:3128"})
    assert out["HTTPS_PROXY"] and out["HTTP_PROXY"]                       # still passed: the warning is the control
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("HTTPS_PROXY" in w for w in warnings) and not any("HTTP_PROXY" in w.replace("HTTPS_PROXY", "") for w in warnings)
    assert "s3cr3t" not in caplog.text and "alice" not in caplog.text


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX only: environment names are case-sensitive there")
def test_posix_matching_is_case_sensitive():
    out = child_env(environ={"path": "/x", "tz": "UTC", "Lang": "en", "lc_all": "C", "TZ": "UTC", "LC_ALL": "C.UTF-8"})
    assert set(out) == {"PATH", "TZ", "LC_ALL"} and out["LC_ALL"] == "C.UTF-8"


def test_the_windows_todo_marker_only_absorbs_assertion_and_key_errors():
    from tests._native import windows_todo
    kw = windows_todo("x").kwargs
    assert kw["strict"] is True and set(kw["raises"]) == {AssertionError, KeyError}

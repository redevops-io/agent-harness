import pytest
from agent_harness.guardrails import redact, validate_input, check_loop

def test_redact_pii():
    text = "Contact test@example.com and key ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    redacted = redact(text)
    assert "[REDACTED_EMAIL]" in redacted
    assert "[REDACTED_TOKEN]" in redacted

def test_max_token_guard():
    assert validate_input("word " * 10, max_tokens=5) is False
    assert validate_input("word " * 3, max_tokens=5) is True

def test_loop_guard():
    # reset not exposed; just exercise
    assert check_loop() is True

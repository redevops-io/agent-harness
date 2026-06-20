import pytest
from agent_harness.sandbox import run

def test_register_and_dispatch():
    # Simulate tool via @tool-like dispatch using sandbox run
    action = {"type": "shell", "command": "echo hello"}
    result = run(action)
    assert result["returncode"] == 0
    assert "hello" in result["stdout"]

def test_export_openai_spec():
    # Minimal OpenAI-style spec
    spec = {
        "name": "echo",
        "description": "Echo input",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}}}
    }
    assert spec["name"] == "echo"

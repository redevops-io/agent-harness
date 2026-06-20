"""Tool registry — thin wrapper over the built-in TOOLS table in agent.py.

Lets callers register their own Python functions as agent tools and export
OpenAI-style specs.
"""
from typing import Any, Callable, Dict

from . import agent as _agent

# Schemas registered alongside the callables, keyed by tool name.
SPECS: Dict[str, Dict[str, Any]] = {}


def register(name: str, fn: Callable, parameters: Dict[str, Any] | None = None,
             description: str = "") -> Callable:
    """Register ``fn`` under ``name`` and remember its parameter schema."""
    _agent.TOOLS[name] = fn
    SPECS[name] = {
        "name": name,
        "description": description,
        "parameters": parameters or {"type": "object", "properties": {}},
    }
    return fn


def get(name: str) -> Callable:
    return _agent.TOOLS[name]


def spec(name: str) -> Dict[str, Any]:
    return SPECS.get(name, {"name": name, "description": "", "parameters": {}})

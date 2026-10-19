"""agent_harness — OSS core for a safe, tool-using agent.

Submodules: agent, command_policy, tools (tool contract + approval gate), guardrails, sandbox, evals, tools, llm, nim,
guardrails_nemo, openshell, h2 (Harness² file-tree spec, validators, artifacts).
"""
from . import (agent, command_policy, guardrails, sandbox, evals, tools, llm, nim,
               guardrails_nemo, openshell, h2)

# Convenient top-level aliases for the real symbols.
from .agent import Agent, create_agent
from .command_policy import CommandPolicy
from .tools import ApprovalPolicy, ToolRegistry, ToolResult, ToolSpec, function_tool

__all__ = [
    "agent",
    "command_policy",
    "guardrails",
    "sandbox",
    "evals",
    "tools",
    "llm",
    "nim",
    "guardrails_nemo",
    "openshell",
    "h2",
    "Agent",
    "create_agent",
    "ApprovalPolicy",
    "CommandPolicy",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "function_tool",
]

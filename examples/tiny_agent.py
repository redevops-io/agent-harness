#!/usr/bin/env python3
"""Tiny self-contained agent example using agent_harness."""

from agent_harness import llm, sandbox
from agent_harness.tools import ToolRegistry, ToolResult, function_tool

def add(args: dict) -> ToolResult:
    return ToolResult(ok=True, data=args["a"] + args["b"], text=str(args["a"] + args["b"]))

def main():
    registry = ToolRegistry()  # default policy: side-effecting tools are denied unless approved
    registry.register(function_tool("add", add, description="Add two integers", parameters={
        "type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}}))
    client = llm.Client()
    with sandbox.Sandbox() as sb:
        result = client.chat("What is 2+3?", tools=registry.list(), sandbox=sb)
        print(result)
        print(registry.run("add", {"a": 2, "b": 3}).text)

if __name__ == "__main__":
    main()

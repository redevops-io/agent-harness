#!/usr/bin/env python3
"""Tiny self-contained agent example using agent_harness."""

from agent_harness import llm, tools, approval, sandbox

def add(a: int, b: int) -> int:
    return a + b

def main():
    tools.register("add", add, {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}})
    client = llm.Client()
    approver = approval.Approver(mode="never")
    with sandbox.Sandbox() as sb:
        result = client.chat("What is 2+3?", tools=["add"], approver=approver, sandbox=sb)
        print(result)

if __name__ == "__main__":
    main()

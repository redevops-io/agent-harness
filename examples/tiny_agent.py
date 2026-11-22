#!/usr/bin/env python3
"""Tiny self-contained agent example using agent_harness."""

from agent_harness import llm, sandbox
from agent_harness.tools import ToolRegistry, ToolResult, function_tool, local_trusted_authorizer

def add(args: dict) -> ToolResult:
    return ToolResult(ok=True, data=args["a"] + args["b"], text=str(args["a"] + args["b"]))

def main():
    # Two gates run on every call. The approval policy denies side effects unless approved, and the
    # authorizer answers "may this caller run this tool" — a registry with no authorizer denies everything,
    # so a standalone example says out loud that it is running under local trust. Behind a service, install
    # the platform's authorizer instead: ToolRegistry(authorizer=my_authz.authorize).
    registry = ToolRegistry(authorizer=local_trusted_authorizer(), classify_required=True)
    # ``action`` is the tool's risk class: which capability the call will be checked for. With no policy
    # plane in the picture it changes nothing here, but declaring it is what lets the same registry be
    # mounted behind a real authorizer without anyone having to guess. A tool that touches nothing is
    # classified "read" — the weakest class there is; there is no "harmless" grade to reach for.
    registry.register(function_tool("add", add, description="Add two integers", parameters={
        "type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}}, action="read"))
    registry.validate(strict=True)      # a boot-time check: classified tools, installed authorizer
    client = llm.Client()
    with sandbox.Sandbox() as sb:
        result = client.chat("What is 2+3?", tools=registry.list(), sandbox=sb)
        print(result)
        print(registry.run("add", {"a": 2, "b": 3}).text)

if __name__ == "__main__":
    main()

"""Context exhaustion: the reusable failure pattern, and the rule that prevents it.

Pattern (G1 run 1, 2026-11-26): a call put 61,716 tokens of retrieved input into a 65,536-token context. The room left for the reply, 3,820 tokens, was spent on reasoning, so the answer
came back EMPTY with ``finish_reason=length``, every time, for the same inputs. Nothing was malformed and nothing timed out; retrying the same request cannot help, and a score of 0 or a
generic "no reply" hides the cause. It cost a seven-hour run.

Rule: execution planning never allocates the whole context to input. The input budget is ``window - reserve``, where the reserve covers the reasoning AND the reply the step needs. A step
whose input cannot fit its budget is refused or reduced (fewer or smaller sources, a longer-context server), never sent with whatever room is left.

This module is the shared arithmetic; a planner calls ``input_budget`` before assembling context, and a runner calls ``exhausted`` to name the failure instead of reporting "no reply".
"""
from __future__ import annotations

from typing import Optional


def input_budget(window: int, reply_tokens: int, reasoning_tokens: int = 0, safety_tokens: int = 256) -> int:
    """The most input a step may carry: the window minus the reasoning and reply it needs (and a small margin for template and tool tokens). Never negative."""
    return max(0, int(window) - int(reply_tokens) - int(reasoning_tokens) - int(safety_tokens))


def reply_room(window: int, prompt_tokens: int) -> int:
    """What is left for reasoning plus reply once the prompt is in; a server clamps the generation to this."""
    return max(0, int(window) - int(prompt_tokens))


def exhausted(prompt_tokens: Optional[int], completion_tokens: Optional[int], finish_reason: str, content: str, window: int, slack: int = 8) -> bool:
    """True for the signature of context exhaustion: generation stopped for length, produced no usable content, and used (nearly) all the room the prompt left."""
    if finish_reason != "length" or (content or "").strip() or prompt_tokens is None or completion_tokens is None:
        return False
    return completion_tokens + slack >= reply_room(window, prompt_tokens)


def fits(prompt_tokens: int, window: int, reply_tokens: int, reasoning_tokens: int = 0, safety_tokens: int = 256) -> bool:
    return prompt_tokens <= input_budget(window, reply_tokens, reasoning_tokens, safety_tokens)

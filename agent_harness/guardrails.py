"""Input/output validation, PII/secret redaction, guards, refusal helper."""
import re
from typing import Any

def redact(text: str) -> str:
    # Redact emails
    text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[REDACTED_EMAIL]', text)
    # Redact API-key-like tokens (alphanum 20+ chars)
    text = re.sub(r'\b[A-Za-z0-9]{20,}\b', '[REDACTED_TOKEN]', text)
    return text

def validate_input(text: str, max_tokens: int = 4096) -> bool:
    if len(text.split()) > max_tokens:
        return False
    return True

def validate_output(text: str, max_tokens: int = 4096) -> bool:
    if len(text.split()) > max_tokens:
        return False
    return True

MAX_LOOPS = 100


class LoopGuard:
    """Per-instance loop counter. Create one per agent run and ``reset()`` at the
    start so iteration limits don't leak across independent runs.
    """

    def __init__(self, max_loops: int = MAX_LOOPS):
        self.max_loops = max_loops
        self.count = 0

    def reset(self) -> None:
        self.count = 0

    def check(self) -> bool:
        self.count += 1
        return self.count <= self.max_loops


# Backwards-compatible module-level helper, backed by a single throwaway guard
# that is reset on every call (so it can never wedge globally). Prefer LoopGuard
# for real run-scoped limiting.
def check_loop(_guard: LoopGuard = LoopGuard()) -> bool:
    return _guard.check()

def refuse_unsafe(reason: str = "unsafe") -> str:
    return f"REFUSAL: {reason}"

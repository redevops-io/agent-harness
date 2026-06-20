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

_loop_counter = 0
MAX_LOOPS = 100

def check_loop() -> bool:
    global _loop_counter
    _loop_counter += 1
    if _loop_counter > MAX_LOOPS:
        return False
    return True

def refuse_unsafe(reason: str = "unsafe") -> str:
    return f"REFUSAL: {reason}"

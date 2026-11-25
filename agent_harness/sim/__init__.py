# SPDX-License-Identifier: AGPL-3.0-or-later
"""Self-Improvement Memory for the harness optimizer, and the proposer-evaluator-selector loop that uses it.

This is memory OF THE OPTIMIZER (what was tried on the harness, why, what was predicted and what happened), not memory
of the agent that solves tasks. After "Self-Improvement Memory Enables Continual Recursive Self-Improvement"
(arXiv:2611.18856), sections 3.1-3.3 and appendix D; the owner's specification is docs/specs/SELF_IMPROVEMENT_MEMORY_SPEC.md.
"""
from .memory import (Attempt, MemoryCondition, PendingAttempt, Problem, Provenance, SIMError, SelfImprovementMemory)

__all__ = ["Attempt", "MemoryCondition", "PendingAttempt", "Problem", "Provenance", "SIMError", "SelfImprovementMemory"]

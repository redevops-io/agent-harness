"""Harness² spec, validators and artifacts (stdlib only; no network, no model calls).

This package defines the eight-component harness file tree, validates it, and stores immutable versioned
copies. It does not run the refinement pipeline: that is upstream google-research/harness2, to be wired
as an optional adapter against a reviewed commit. See docs/harness2.md.
"""
from .artifact import ArtifactError, lineage, load_artifact, save_artifact, tree_hash
from .spec import CODEX, OPENCODE, SURFACES, UPSTREAM_COMMIT, HarnessSpec, Surface, stock_spec
from .validate import Issue, validate

__all__ = [
    "OPENCODE", "CODEX", "SURFACES", "Surface", "UPSTREAM_COMMIT", "HarnessSpec", "stock_spec",
    "Issue", "validate",
    "ArtifactError", "save_artifact", "load_artifact", "tree_hash", "lineage",
]

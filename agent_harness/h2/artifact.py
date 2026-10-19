"""Immutable, content-addressed harness artifacts with lineage.

An artifact directory is ``<root>/<sha256>/`` holding ``tree/`` and ``manifest.json``. The id is the hash
of the tree, so identical trees collapse to one artifact and a stored artifact can never change silently.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from .spec import HarnessSpec, normalize
from .validate import validate


class ArtifactError(Exception):
    pass


def tree_hash(spec: HarnessSpec) -> str:
    h = hashlib.sha256()
    for path in sorted(spec.files):
        data = spec.files[path].encode("utf-8")
        h.update(f"{len(path)}:{path}\0{len(data)}:".encode())
        h.update(data)
    return h.hexdigest()


def save_artifact(root: Path, spec: HarnessSpec, *, parent: Optional[str] = None,
                  meta: Optional[Dict[str, Any]] = None) -> str:
    """Store ``spec`` under ``root`` and return its id. Refuses invalid trees and unknown parents."""
    issues = validate(spec)
    if issues:
        raise ArtifactError("invalid harness tree:\n" + "\n".join(str(i) for i in issues))
    root = Path(root)
    if parent is not None and not (root / parent / "manifest.json").is_file():
        raise ArtifactError(f"unknown parent artifact {parent!r}")
    aid = tree_hash(spec)
    dest = root / aid
    if dest.exists():
        verify_artifact(root, aid)
        return aid
    tmp = root / f".tmp-{aid}"
    if tmp.exists():
        shutil.rmtree(tmp)
    for path, text in spec.files.items():
        f = tmp / "tree" / path
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text, encoding="utf-8")
    manifest = {"id": aid, "parent": parent, "files": sorted(spec.files), "meta": meta or {}}
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    tmp.rename(dest)
    return aid


def load_artifact(root: Path, aid: str) -> HarnessSpec:
    """Load and verify an artifact; raises ArtifactError if the stored tree no longer matches its id."""
    tree = Path(root) / aid / "tree"
    if not tree.is_dir():
        raise ArtifactError(f"no such artifact {aid!r}")
    spec = HarnessSpec()
    for f in sorted(p for p in tree.rglob("*") if p.is_file()):
        if f.is_symlink():
            raise ArtifactError(f"symlink in artifact: {f}")
        spec.set(normalize(f.relative_to(tree).as_posix()), f.read_text(encoding="utf-8"))
    if tree_hash(spec) != aid:
        raise ArtifactError(f"artifact {aid} fails hash verification (tree was modified)")
    return spec


def verify_artifact(root: Path, aid: str) -> None:
    load_artifact(root, aid)


def lineage(root: Path, aid: str) -> list:
    """Return ids from ``aid`` back to its root ancestor."""
    chain, seen = [], set()
    while aid is not None:
        if aid in seen:
            raise ArtifactError("cycle in artifact lineage")
        seen.add(aid)
        mpath = Path(root) / aid / "manifest.json"
        if not mpath.is_file():
            raise ArtifactError(f"missing manifest for {aid}")
        chain.append(aid)
        aid = json.loads(mpath.read_text(encoding="utf-8")).get("parent")
    return chain

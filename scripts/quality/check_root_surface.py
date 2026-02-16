#!/usr/bin/env python3
"""Validate the tracked root-level file surface against an allowlist."""

from __future__ import annotations

import subprocess
from pathlib import Path


def _load_allowlist(path: Path) -> set[str]:
    allowed: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        allowed.add(line)
    return allowed


def _tracked_root_files(repo_root: Path) -> set[str]:
    proc = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    tracked = set()
    for rel_path in proc.stdout.splitlines():
        rel_path = rel_path.strip()
        if not rel_path:
            continue
        if "/" not in rel_path and (repo_root / rel_path).is_file():
            tracked.add(rel_path)
    return tracked


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    allowlist_path = repo_root / "scripts" / "quality" / "root_surface_allowlist.txt"
    allowed = _load_allowlist(allowlist_path)
    tracked_root = _tracked_root_files(repo_root)

    unexpected = sorted(tracked_root - allowed)
    missing = sorted(allowed - tracked_root)

    if not unexpected and not missing:
        print(f"[root-surface] ok ({len(tracked_root)} root files tracked)")
        return 0

    print("[root-surface] mismatch detected")
    if unexpected:
        print("unexpected tracked root files:")
        for name in unexpected:
            print(f"  - {name}")
    if missing:
        print("allowlisted root files not currently tracked:")
        for name in missing:
            print(f"  - {name}")
    print("update scripts/quality/root_surface_allowlist.txt when changing root surface intentionally.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

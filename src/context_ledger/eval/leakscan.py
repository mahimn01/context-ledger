"""leakscan (B.4): look for the API key everywhere it must never be. Read-only.

Layer 1, runtime artifacts, for the BARE prefix "sk-ant" (nothing legitimate ever writes it
there): runs/, .clp-spend/, every *.sqlite in the repo, and .cache/corpus/. The builder's own
build*.log files are checked for the key shape only.
Layer 2, the tracked tree and `git log -p --all`, for the key SHAPE sk-ant-[A-Za-z0-9_-]{20,}
(the bare prefix legitimately appears in the brief, the quoted rule and the redaction regex).
.venv/ and the CPython tarball are excluded.
With a Secret still in memory (at the end of a live run) the exact key bytes are looked for in
every runtime artifact too. Only paths and counts are ever reported, never matched content.
"""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BARE = b"sk-" + b"ant"
KEY_SHAPE = re.compile(rb"sk-" + rb"ant-[A-Za-z0-9_-]{20,}")
EXCLUDED_DIRS = (".venv", ".git", "__pycache__", ".pytest_cache", ".hypothesis")


@dataclass
class LeakReport:
    findings: list[tuple[str, str, int]] = field(default_factory=list)  # (path, what, count)
    scanned: int = 0

    @property
    def ok(self) -> bool:
        return not self.findings

    def render(self) -> str:
        lines = [f"leakscan: scanned {self.scanned} files and the git history: {'PASS' if self.ok else 'FAIL'}"]
        lines += [f"  {path}: {count} {what}" for path, what, count in self.findings]
        return "\n".join(lines)


def _files(root: Path) -> Iterable[Path]:
    """Every file under `root` (or `root` itself), skipping EXCLUDED_DIRS without entering them."""
    if root.is_file():
        yield root
        return
    for folder, dirs, names in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDED_DIRS)
        for name in sorted(names):
            yield Path(folder) / name


def runtime_paths(root: Path = ROOT) -> list[Path]:
    """Every runtime artifact under the repo: runs/, .clp-spend/, .cache/corpus/ and *.sqlite."""
    paths = {*_files(root / "runs"), *_files(root / ".clp-spend"), *_files(root / ".cache" / "corpus")}
    paths |= {p for p in _files(root) if p.suffix in (".sqlite",) or p.name.endswith((".sqlite-wal", ".sqlite-shm"))}
    return sorted(paths)


def scan(
    *,
    root: Path = ROOT,
    runtime: Iterable[Path] | None = None,
    secret=None,
    git: bool = True,
) -> LeakReport:
    """Run both layers (see the module docstring). `runtime` overrides the runtime artifacts to
    scan (tests point it at their temporary run directories)."""
    report = LeakReport()
    for path in runtime if runtime is not None else runtime_paths(root):
        for file in _files(Path(path)):
            data = file.read_bytes()
            report.scanned += 1
            if data.count(BARE):
                report.findings.append((str(file), "bare sk-ant prefix", data.count(BARE)))
            if secret is not None and secret.occurs_in(data):
                report.findings.append((str(file), "exact key bytes", 1))
    for log in sorted(root.glob("build*.log")):
        report.scanned += 1
        hits = len(KEY_SHAPE.findall(log.read_bytes()))
        if hits:
            report.findings.append((str(log), "key-shaped string", hits))
    if git and (root / ".git").exists():
        listed = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=root, capture_output=True, check=True)
        for name in listed.stdout.decode("utf-8", errors="replace").splitlines():
            path = root / name
            if not path.is_file() or name.endswith(".tgz"):
                continue
            report.scanned += 1
            hits = len(KEY_SHAPE.findall(path.read_bytes()))
            if hits:
                report.findings.append((name, "key-shaped string", hits))
        history = subprocess.run(["git", "log", "-p", "--all"], cwd=root, capture_output=True, check=True).stdout
        hits = len(KEY_SHAPE.findall(history))
        if hits:
            report.findings.append(("git log -p --all", "key-shaped string", hits))
    return report

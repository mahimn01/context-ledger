"""The CPython corpus builder (C.1): nested corpora C10 ⊂ C100 ⊂ C1000 and their FTS5 indexes.

Source. The official python.org source release v3.15.0 at .cache/cpython/Python-3.15.0.tgz. Its
SHA-256 is verified before anything is read, and it is read with `tarfile`, never extracted.
Files: .py, .rst, .txt, .c, .h and .md under the release directory, UTF-8 decodable, at most
512 KB each.

Sizes. The window is W = 6,000 tokens (PREREG section 2; M3 used 16,000) and rho in {10, 100,
1000}, so the targets are 60k, 600k and 6M tokens, measured with the estimator in force (calib-v1 once calibrated, else bytes4-v0,
with a warning in the manifest). Nesting: the gold files come first (the .py files questions
are asked about), then every other file in a fixed seeded order with .py files before the other
types, so the small corpora are code-dominant. Whole files are added until each target is
reached; a shortfall fails loudly.

Output. runs/corpus/<id>/manifest.json (release, sha256, seed, every file with its size, token
count and the sha256 of its perturbed text, the totals and the estimator) and
runs/corpus/<id>/perturbations.jsonl; the perturbed texts go to .cache/corpus/<id>/files/ and
the index to .cache/corpus/<id>/index.sqlite (both gitignored). The id is a hash of the inputs.

Index. One SQLite FTS5 table per rho, `chunks_<rho>`, over chunks of that corpus's files: .py
files are cut at top-level defs and classes, other files into fixed 60-line windows, and any
chunk longer than 60 lines is split into 60-line windows. Ranked by bm25(). FTS5 must be
available; there is no silent fallback.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import sqlite3
import sys
import tarfile
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..ids import canonical_json, sha256_hex
from ..tokens import BYTES4_V0, source_class
from . import pyscan
from .perturb import perturb_corpus

ROOT = Path(__file__).resolve().parents[3]
RELEASE = "v3.15.0"
TARBALL = ROOT / ".cache" / "cpython" / "Python-3.15.0.tgz"
TARBALL_SHA256 = "438596cac081036d3c1d532ab7e7335eeb35567bc961749a0d5797176db0db68"
CACHE_DIR = ROOT / ".cache" / "corpus"
RUNS_CORPUS = ROOT / "runs" / "corpus"
BUILDER = "corpus-v2"  # v2 (M4): gold files ordered by question-candidate density
BUILDER_V2 = "corpus-v3"  # `corpus build --version v2` (BUILD-V2): v2 question density, hop files after their gold file
BUILDERS = {"v1": BUILDER, "v2": BUILDER_V2}
EXTENSIONS = (".py", ".rst", ".txt", ".c", ".h", ".md")
MAX_FILE_BYTES = 512 * 1024
WINDOW = 6000  # PREREG section 2: at 16,000 no fold or compaction fired in the live smoke run
RHOS = (10, 100, 1000)
CHUNK_LINES = 60
N_GOLD = 60
GOLD_BYTES = (2 * 1024, 16 * 1024)
_TEST_DIR = re.compile(r"(^|/)(test|tests|idle_test)(/|$)")


class CorpusError(Exception):
    """The corpus cannot be built as specified (bad tarball, no FTS5, too little text)."""


def require_fts5() -> None:
    """Raise CorpusError, with a clear message, if this SQLite has no FTS5."""
    try:
        with sqlite3.connect(":memory:") as db:
            db.execute("CREATE VIRTUAL TABLE probe USING fts5(body)")
    except sqlite3.OperationalError as exc:
        raise CorpusError(f"this Python's SQLite ({sqlite3.sqlite_version}) has no FTS5 ({exc}); the RAG index needs it") from None


def verify_tarball(path: str | Path, sha256: str) -> None:
    """Refuse a tarball whose SHA-256 is not the expected one."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != sha256:
        raise CorpusError(f"{path} has sha256 {digest.hexdigest()}, expected {sha256}; refusing to use it")


def read_sources(path: str | Path = TARBALL, sha256: str = TARBALL_SHA256) -> list[tuple[str, str]]:
    """(path relative to the release directory, text) of every eligible file, sorted by path.
    The tarball is verified first and read in memory, never extracted."""
    verify_tarball(path, sha256)
    out = []
    with tarfile.open(path, "r:gz") as tar:
        for member in tar:
            parts = member.name.split("/", 1)
            if not member.isfile() or len(parts) != 2 or not parts[1].endswith(EXTENSIONS) or member.size > MAX_FILE_BYTES:
                continue
            data = tar.extractfile(member).read()
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                continue
            out.append((parts[1], text.replace("\r\n", "\n").replace("\r", "\n")))
    out.sort()
    return out


def is_gold_candidate(path: str, text: str) -> bool:
    """A library module (not a test) of 2-16 KB that parses and has at least two functions and
    two literal sites to perturb: a file questions can be asked about."""
    size = len(text.encode("utf-8"))
    if not (path.startswith("Lib/") and path.endswith(".py")) or _TEST_DIR.search(path) or not GOLD_BYTES[0] <= size <= GOLD_BYTES[1]:
        return False
    tree = pyscan.parse(text)
    return tree is not None and len(pyscan.functions(tree)) >= 2 and len(pyscan.literal_sites(tree)) >= 2


def order_files(files: Sequence[tuple[str, str]], seed: int, n_gold: int = N_GOLD, tokens: dict[str, int] | None = None) -> tuple[list[str], list[str]]:
    """(gold paths, every path in corpus order): the gold files first, then the rest in a
    seeded order with .py files before the other types.

    Gold candidates are ordered by question-candidate density, most questions per token first
    (questions.question_density on the ORIGINAL text, filters included, no perturbed value
    used), ties broken by a seeded shuffle. The smallest corpus then holds files questions can
    be asked about: at W = 6,000 calib-v1 tokens C10 has room for only about ten files, and a
    seeded order alone filled half of it with codec tables whose methods are defined in ~100
    files, so no question passes the unique-definition filter. `tokens` is each file's count
    with the estimator in force (bytes/4 if not given)."""
    from .questions import defined_counts, question_density  # questions imports this module

    rng = random.Random(f"corpus:{seed}")
    candidates = [p for p, t in files if is_gold_candidate(p, t)]
    rng.shuffle(candidates)
    texts = dict(files)
    defined = defined_counts(texts)
    size = tokens or {p: (len(t.encode("utf-8")) + 3) // 4 for p, t in files}
    density = {p: question_density(p, texts[p], defined, size[p]) for p in candidates}
    candidates.sort(key=lambda p: -density[p])  # stable: the seeded shuffle breaks ties
    gold = candidates[:n_gold]
    rest = [p for p, _ in files if p not in set(gold)]
    py = [p for p in rest if p.endswith(".py")]
    other = [p for p in rest if not p.endswith(".py")]
    rng.shuffle(py)
    rng.shuffle(other)
    return gold, gold + py + other


def order_files_v2(files: Sequence[tuple[str, str]], seed: int, n_gold: int = N_GOLD, tokens: dict[str, int] | None = None) -> tuple[list[str], list[str]]:
    """The v2 corpus order. Gold candidates with multi-hop questions come first, then the rest by
    v2 question density (questions_v2.density_v2 on the ORIGINAL text: v1's templates with
    confusable subjects plus the multi-hop ones), ties by a seeded shuffle. Multi-hop questions
    are rare in CPython's small library modules (6 in 5 of its 136 gold candidates), so they go
    first: then whatever the corpus supplies is in the smallest corpus. The ranking uses the
    templates only, never an answer. Each gold file is followed by the files its cross-file
    multi-hop questions resolve into (so every hop file is in the smallest corpus with its gold
    file); then the rest in a seeded order, .py files first."""
    from .questions import defined_counts
    from .questions_v2 import ModuleIndex, density_v2

    rng = random.Random(f"corpus-v2:{seed}")
    candidates = [p for p, t in files if is_gold_candidate(p, t)]
    rng.shuffle(candidates)
    texts = dict(files)
    defined = defined_counts(texts)
    index = ModuleIndex(texts)
    size = tokens or {p: (len(t.encode("utf-8")) + 3) // 4 for p, t in files}
    scored = {p: density_v2(p, texts[p], index, defined, size[p]) for p in candidates}
    candidates.sort(key=lambda p: (scored[p][0] == 0, -scored[p][1]))
    gold = candidates[:n_gold]
    order: list[str] = []
    for path in gold:
        order += [q for q in (path, *scored[path][2]) if q not in order]
    placed = set(order)
    rest = [p for p, _ in files if p not in placed]
    py = [p for p in rest if p.endswith(".py")]
    other = [p for p in rest if not p.endswith(".py")]
    rng.shuffle(py)
    rng.shuffle(other)
    return gold, order + py + other


def cut_points(order: Sequence[str], tokens: dict[str, int], targets: dict[int, int]) -> dict[int, int]:
    """For each rho, the number of leading files whose total first reaches its target. Raises
    CorpusError with the shortfall if the files run out first."""
    cuts, total, n = {}, 0, 0
    for rho in sorted(targets):
        while total < targets[rho] and n < len(order):
            total += tokens[order[n]]
            n += 1
        if total < targets[rho]:
            raise CorpusError(f"C{rho} needs {targets[rho]} tokens; the eligible files hold only {total} (short by {targets[rho] - total})")
        cuts[rho] = n
    return cuts


def chunk_lines(path: str, text: str) -> list[tuple[int, int]]:
    """1-based inclusive (start, end) line ranges covering the file: .py files cut at each
    top-level def and class (decorators included), other files in fixed windows; every chunk
    is at most CHUNK_LINES long."""
    n = text.count("\n") + (0 if text.endswith("\n") else 1)
    if n == 0:
        return []
    starts = [1]
    if path.endswith(".py"):
        tree = pyscan.parse(text)
        starts += [s for s in (pyscan.top_level_starts(tree) if tree else []) if s > 1]
    starts = sorted(set(starts))
    out = []
    for k, start in enumerate(starts):
        end = starts[k + 1] - 1 if k + 1 < len(starts) else n
        for s in range(start, end + 1, CHUNK_LINES):
            out.append((s, min(end, s + CHUNK_LINES - 1)))
    return out


def lines_of(text: str, start: int, end: int) -> str:
    """Lines start..end (1-based, inclusive) of `text`, joined with newlines."""
    return "\n".join(text.split("\n")[start - 1 : end])


_QUERY_TOKEN = re.compile(r"[A-Za-z0-9]+")


def fts_query(text: str) -> str | None:
    """An FTS5 MATCH expression OR-ing the distinct alphanumeric tokens of `text`, or None."""
    tokens = list(dict.fromkeys(t.lower() for t in _QUERY_TOKEN.findall(text)))
    return " OR ".join(f'"{t}"' for t in tokens) if tokens else None


@dataclass(frozen=True)
class Hit:
    path: str
    start: int
    end: int
    snippet: str  # the chunk's first non-blank line, at most 100 characters


class CorpusIndex:
    """BM25 search over one rho's chunks, and line reads of the perturbed files."""

    def __init__(self, directory: str | Path, rho: int) -> None:
        self.directory = Path(directory)
        self.rho = rho
        self.db = sqlite3.connect(self.directory / "index.sqlite")
        self.files = self.directory / "files"
        self._texts: dict[str, str] = {}
        self._bm25: tuple[int, float] | None = None
        if not self.db.execute("SELECT name FROM sqlite_master WHERE name = ?", (f"chunks_{rho}",)).fetchone():
            raise CorpusError(f"no index for rho={rho} in {self.directory}")
        self.paths = {row[0] for row in self.db.execute(f"SELECT DISTINCT path FROM chunks_{rho}")}

    def text(self, path: str) -> str:
        if path not in self.paths:
            raise KeyError(path)
        if path not in self._texts:
            self._texts[path] = (self.files / path).read_text(encoding="utf-8")
        return self._texts[path]

    def search(self, query: str, limit: int = 10) -> list[Hit]:
        """Top chunks by bm25 (ties by insertion order, which is deterministic)."""
        return [hit for _, hit in self.scored_search(query, limit)]

    def scored_search(self, query: str, limit: int = 10) -> list[tuple[float, Hit]]:
        """`search` with each chunk's FTS5 bm25() score (lower is better)."""
        match = fts_query(query)
        if match is None:
            return []
        rows = self.db.execute(
            f"SELECT path, start, end, body, bm25(chunks_{self.rho}) FROM chunks_{self.rho} WHERE chunks_{self.rho} MATCH ? "
            f"ORDER BY bm25(chunks_{self.rho}), rowid LIMIT ?",
            (match, limit),
        ).fetchall()
        return [(score, Hit(p, s, e, next((line.strip() for line in body.split("\n") if line.strip()), "")[:100])) for p, s, e, body, score in rows]

    def bm25(self, query: str, text: str) -> float | None:
        """FTS5's bm25() of `text` for `query`, scored as one more document of this index against
        the index's own statistics (row count, average length, document frequencies; k1 1.2,
        b 0.75, idf floored at 1e-6, as fts5_aux.c computes it), so it ranks among
        `scored_search` scores (equal to within floating-point rounding); None if no query
        term occurs in `text`. BUILD-V2c: B3 retrieves an episode's notes this way. The
        statistics come from an fts5vocab table in the temp schema: the index is not written."""
        terms = list(dict.fromkeys(t.lower() for t in _QUERY_TOKEN.findall(query)))
        words = [t.lower() for t in _QUERY_TOKEN.findall(text)]
        if not any(t in words for t in terms):
            return None
        rows, avgdl = self._stats()
        score = 0.0
        for term in terms:
            tf = words.count(term)
            if tf:
                df = self.db.execute(f"SELECT doc FROM temp.vocab_{self.rho} WHERE term = ?", (term,)).fetchone()
                n = df[0] if df else 0
                idf = math.log((0.5 + rows - n) / (0.5 + n))
                score += (idf if idf > 0 else 1e-6) * (tf * 2.2) / (tf + 1.2 * (1 - 0.75 + 0.75 * len(words) / avgdl))
        return -score

    def _stats(self) -> tuple[int, float]:
        if self._bm25 is None:
            self.db.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS temp.vocab_{self.rho} USING fts5vocab(main, chunks_{self.rho}, 'row')")
            rows = self.db.execute(f"SELECT count(*) FROM chunks_{self.rho}").fetchone()[0]
            tokens = self.db.execute(f"SELECT coalesce(sum(cnt), 0) FROM temp.vocab_{self.rho}").fetchone()[0]
            self._bm25 = (rows, tokens / rows if rows else 1.0)
        return self._bm25

    def close(self) -> None:
        self.db.close()


def _progress(message: str) -> None:
    sys.stderr.write(message + "\n")
    sys.stderr.flush()


def corpus_id(inputs: dict) -> str:
    return "c" + sha256_hex(canonical_json(inputs))[:12]


def build_index(directory: Path, texts: dict[str, str], order: Sequence[str], cuts: dict[int, int], progress: Callable[[str], None]) -> None:
    """(Re)write index.sqlite with one FTS5 table per rho over that corpus's chunks."""
    require_fts5()
    db_path = directory / "index.sqlite"
    if db_path.exists():
        db_path.unlink()
    with sqlite3.connect(db_path) as db:
        for rho, n in sorted(cuts.items()):
            started = time.monotonic()
            db.execute(f"CREATE VIRTUAL TABLE chunks_{rho} USING fts5(path UNINDEXED, start UNINDEXED, end UNINDEXED, body)")
            rows = ((p, s, e, lines_of(texts[p], s, e)) for p in order[:n] for s, e in chunk_lines(p, texts[p]))
            db.executemany(f"INSERT INTO chunks_{rho} (path, start, end, body) VALUES (?, ?, ?, ?)", rows)
            count = db.execute(f"SELECT count(*) FROM chunks_{rho}").fetchone()[0]
            progress(f"indexed C{rho}: {n} files, {count} chunks in {time.monotonic() - started:.1f}s")


def build_corpus(
    *,
    seed: int,
    estimator=BYTES4_V0,
    estimator_warning: str | None = None,
    tarball: str | Path = TARBALL,
    sha256: str = TARBALL_SHA256,
    window: int = WINDOW,
    rhos: Iterable[int] = RHOS,
    n_gold: int = N_GOLD,
    runs_dir: str | Path = RUNS_CORPUS,
    cache_dir: str | Path = CACHE_DIR,
    perturb: Callable | None = perturb_corpus,
    progress: Callable[[str], None] = _progress,
    version: str = "v1",
) -> dict:
    """Build the nested corpora and their indexes; return the manifest (also written to disk).

    `perturb(files, order, gold, seed)` (perturb.perturb_corpus by default) returns (perturbed
    texts by path, edit records, excluded paths); the corpus then holds the perturbed texts.
    With perturb=None the texts are the originals (tests only). Sizes and cuts are measured
    on the original texts; a perturbation changes a few bytes per edit. `version` "v1" is the M4
    builder (unchanged, so its corpus ids reproduce); "v2" orders files with `order_files_v2`."""
    if version not in BUILDERS:
        raise CorpusError(f"unknown corpus version {version!r}; choose from {sorted(BUILDERS)}")
    started = time.monotonic()
    rhos = tuple(sorted(set(rhos)))
    files = read_sources(tarball, sha256)
    progress(f"read {len(files)} eligible files from {Path(tarball).name} (sha256 verified) in {time.monotonic() - started:.1f}s")
    texts = dict(files)
    tokens = {p: estimator.count(t, source_class(p)) for p, t in files}
    gold, order = (order_files if version == "v1" else order_files_v2)(files, seed, n_gold, tokens)
    targets = {rho: window * rho for rho in rhos}
    cuts = cut_points(order, tokens, targets)
    members = order[: cuts[max(rhos)]]
    inputs = {"builder": BUILDERS[version], "release": RELEASE, "sha256": sha256, "seed": seed, "window": window, "rhos": list(rhos),
              "n_gold": n_gold, "estimator": estimator.payload(), "perturbed": perturb is not None}  # fmt: skip
    cid = corpus_id(inputs)
    edits, excluded = [], []
    if perturb is not None:
        perturbed, edits, excluded = perturb([(p, texts[p]) for p in members], members, gold, seed)
        texts.update(perturbed)
        progress(f"perturbed {len({e['path'] for e in edits})} files with {len(edits)} edits; {len(excluded)} reverted")
    out_dir, cache = Path(runs_dir) / cid, Path(cache_dir) / cid
    files_dir = cache / "files"
    for path in members:
        target = files_dir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(texts[path], encoding="utf-8")
    build_index(cache, texts, order, cuts, progress)
    smallest = {}
    for rho in reversed(rhos):
        for path in order[: cuts[rho]]:
            smallest[path] = rho
    manifest = {
        **inputs,
        "id": cid,
        "estimator_warning": estimator_warning,
        "targets": {str(r): targets[r] for r in rhos},
        "totals": {str(r): sum(tokens[p] for p in order[: cuts[r]]) for r in rhos},
        "counts": {str(r): cuts[r] for r in rhos},
        "gold": gold,
        "excluded": sorted(excluded),
        "files": [
            {"path": p, "bytes": len(texts[p].encode("utf-8")), "tokens": tokens[p], "sha256": sha256_hex(texts[p].encode("utf-8")), "rho": smallest[p]}
            for p in members
        ],
        "cache_dir": str(cache),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    with open(out_dir / "perturbations.jsonl", "w", encoding="utf-8") as f:
        for edit in edits:
            f.write(json.dumps(edit, sort_keys=True, ensure_ascii=False) + "\n")
    progress(f"corpus {cid}: " + ", ".join(f"C{r} {manifest['totals'][str(r)]} tokens in {cuts[r]} files" for r in rhos) + f" ({time.monotonic() - started:.1f}s)")
    return manifest


def load_manifest(corpus: str | Path | None = None, runs_dir: str | Path = RUNS_CORPUS) -> dict:
    """The manifest of corpus id (or directory) `corpus`, or of the most recently built one."""
    root = Path(runs_dir)
    if corpus is not None and Path(corpus).is_dir():
        return json.loads((Path(corpus) / "manifest.json").read_text(encoding="utf-8"))
    if corpus is not None:
        return json.loads((root / str(corpus) / "manifest.json").read_text(encoding="utf-8"))
    found = sorted(root.glob("*/manifest.json"), key=lambda p: p.stat().st_mtime)
    if not found:
        raise CorpusError(f"no corpus in {root}; run `corpus build` first")
    return json.loads(found[-1].read_text(encoding="utf-8"))


def corpus_texts(manifest: dict) -> dict[str, str]:
    """Every perturbed file text of the corpus, by path."""
    files = Path(manifest["cache_dir"]) / "files"
    return {f["path"]: (files / f["path"]).read_text(encoding="utf-8") for f in manifest["files"]}

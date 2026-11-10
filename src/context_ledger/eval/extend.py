"""The nested extension level (BUILD-V4 item 8): a corpus whose C10, C100 and C1000 are a base
corpus's, byte for byte, and whose top level adds open-source Python packages installed on this
machine as distractors only (no question is ever asked about them).

Sources. `freeze_sources` surveys site-packages directories once and writes the frozen source
list (src/context_ledger/eval/ext_sources_v4.json, tracked): one entry per distribution with a
permissive license (MIT, BSD, Apache, PSF, ISC, 0BSD, MIT-CMU, Unlicense, HPND, MPL), each
distribution taken once (the copy in the newest Python's site-packages), with its version,
license, and the sha256 over its eligible files' (path, sha256) pairs. Eligible files are the
ones its RECORD lists with the corpus builder's extensions (.py, .rst, .txt, .c, .h, .md), at most
512 KB, UTF-8 decodable, and for .py files parseable by this interpreter (as the builder reverts
a perturbed file that no longer parses; an unparseable file would also make L's span
registration fail). `extend_corpus` re-reads every listed file and refuses (CorpusError) if a
distribution's digest differs from the frozen one: an upgraded package cannot change the corpus
silently.

Layout and order. An extension file keeps its site-packages path under Lib/site-packages/, where
CPython's own tree has third-party packages (`Lib/site-packages/numpy/core/numeric.py`). A path
two distributions share, or a text another file already has, is taken once (the first in the
frozen list). After the base corpus's files, in its order, come the extension files in a seeded
order (random.Random(f"corpus-ext:{seed}")), .py files first, then the rest, added whole until
the level's target W x rho (W the base's window) is reached, counted with the base's estimator.
If the sources run out first, the level is built with all of them and labelled with the rho it
reaches (tokens / W, rounded down to a power of ten when that is still above the base's top
level, else down to an integer), and the manifest says so.

Perturbation. Extension .py files get the base builder's literal perturbation (perturb.
literal_edits with the base's seed: module constants and literal defaults), never its renames
(those were chosen over the base corpus and stay as they are); a file that would no longer parse
keeps its original text. Edits are appended to the base's records in perturbations.jsonl.

Output: runs/corpus/<id>/manifest.json (the base's fields, `base`, the extension block and every
file with its sha256 and source sha256), perturbations.jsonl, and .cache/corpus/<id>/files/ plus
index.sqlite with chunks_<rho> for every level (the base levels' tables hold the same rows in the
same order as the base's, so BM25 ranks the same). Question files are not copied: a run on the
extension reads a questions version the extension does not have from its base (runner).
"""

from __future__ import annotations

import ast
import json
import random
import sys
import time
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from email.parser import Parser
from pathlib import Path
from typing import NamedTuple

from ..ids import sha256_hex
from ..tokens import calibration_estimator, source_class
from .corpus import CACHE_DIR, EXTENSIONS, MAX_FILE_BYTES, RUNS_CORPUS, CorpusError, build_index, corpus_id
from .perturb import apply_edits, literal_edits

BUILDER_EXT = "corpus-ext-v4"
SOURCES_FILE = Path(__file__).with_name("ext_sources_v4.json")
PREFIX = "Lib/site-packages/"
PERMISSIVE = ("MIT", "BSD", "APACHE", "PYTHON SOFTWARE FOUNDATION", "PSF", "ISC", "0BSD", "UNLICENSE", "HPND", "MPL", "MOZILLA")
NOT_PERMISSIVE = ("GPL", "AGPL", "PROPRIETARY", "COMMERCIAL")


def license_of(metadata_text: str) -> str:
    """The license a METADATA file declares: License-Expression, else the License classifiers,
    else the License field's first line."""
    meta = Parser().parsestr(metadata_text)
    expr = meta.get("License-Expression") or ""
    classifiers = [c.split("::")[-1].strip() for c in (meta.get_all("Classifier") or []) if c.startswith("License ::")]
    field = (meta.get("License") or "").strip().splitlines()[:1]
    return expr or "; ".join(classifiers) or (field[0] if field else "")


def permissive(license_text: str) -> bool:
    up = license_text.upper()
    return bool(up) and any(k in up for k in PERMISSIVE) and not any(k in up for k in NOT_PERMISSIVE)


def _parses(text: str) -> bool:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ast.parse(text)
        return True
    except (SyntaxError, ValueError):
        return False


def text_of(raw: bytes) -> str | None:
    """A source file's text as the corpus takes it (UTF-8, newlines normalized), or None."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return text.replace("\r\n", "\n").replace("\r", "\n")


def eligible_raw(site: Path, dist_info: Path) -> list[tuple[str, bytes]]:
    """(path relative to site-packages, bytes) of a distribution's eligible files (module
    docstring), sorted by path."""
    record = dist_info / "RECORD"
    if not record.exists():
        return []
    out = []
    for line in record.read_text(encoding="utf-8", errors="replace").splitlines():
        rel = line.split(",")[0]
        if not rel.endswith(EXTENSIONS) or rel.startswith(("..", "/")) or ".dist-info/" in rel or "__pycache__" in rel:
            continue
        path = site / rel
        if not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
            continue
        raw = path.read_bytes()
        text = text_of(raw)
        if text is None or (rel.endswith(".py") and not _parses(text)):
            continue
        out.append((rel, raw))
    return sorted(dict(out).items())


def eligible_files(site: Path, dist_info: Path) -> list[tuple[str, str]]:
    """(path relative to site-packages, text) of a distribution's eligible files, sorted by path."""
    return [(rel, text_of(raw)) for rel, raw in eligible_raw(site, dist_info)]


def files_digest(files: Iterable[tuple[str, str]]) -> str:
    return sha256_hex("".join(f"{rel}\t{sha256_hex(text.encode('utf-8'))}\n" for rel, text in sorted(files)).encode("utf-8"))


def _version_key(site: Path) -> tuple:
    name = site.parent.name  # python3.14
    digits = "".join(c if c.isdigit() else " " for c in name.removeprefix("python")).split()
    return tuple(int(d) for d in digits) + ((0,) if name.endswith("t") else (1,))


def freeze_sources(sites: Iterable[str | Path], out: str | Path = SOURCES_FILE, progress: Callable[[str], None] | None = None) -> dict:
    """Survey `sites` (site-packages directories) and write the frozen source list (module
    docstring). Returns it."""
    chosen: dict[str, dict] = {}
    for site in sorted((Path(s) for s in sites), key=_version_key, reverse=True):
        for dist_info in sorted(site.glob("*.dist-info")):
            metadata = dist_info / "METADATA"
            if not metadata.exists():
                continue
            meta = Parser().parsestr(metadata.read_text(encoding="utf-8", errors="replace"))
            name = (meta.get("Name") or dist_info.name.split("-")[0]).lower().replace("_", "-")
            lic = license_of(metadata.read_text(encoding="utf-8", errors="replace"))
            if name in chosen or not permissive(lic):
                continue
            files = eligible_files(site, dist_info)
            if not files:
                continue
            chosen[name] = {"name": name, "version": meta.get("Version"), "license": lic, "site": str(site), "dist_info": dist_info.name,
                            "files": len(files), "bytes": sum(len(t.encode("utf-8")) for _, t in files), "sha256": files_digest(files)}  # fmt: skip
            if progress is not None:
                progress(f"source {name} {meta.get('Version')}: {len(files)} files ({lic})")
    spec = {"version": "ext-sources-v4", "python": f"{sys.version_info[0]}.{sys.version_info[1]}", "distributions": sorted(chosen.values(), key=lambda d: d["name"])}
    Path(out).write_text(json.dumps(spec, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return spec


class Source(NamedTuple):
    """An extension source file: its corpus path, its text, its distribution and its bytes."""

    path: str
    text: str
    dist: str
    raw: bytes


def load_sources(spec: dict) -> list[Source]:
    """Every frozen distribution's files, verified against the frozen digests (CorpusError on
    any difference); a path or a text seen earlier is skipped."""
    if spec.get("version") != "ext-sources-v4":
        raise CorpusError("not an ext-sources-v4 source list")
    out, paths, digests = [], set(), set()
    for dist in spec["distributions"]:
        raw = eligible_raw(Path(dist["site"]), Path(dist["site"]) / dist["dist_info"])
        files = [(rel, text_of(data)) for rel, data in raw]
        if files_digest(files) != dist["sha256"] or len(files) != dist["files"]:
            raise CorpusError(f"distribution {dist['name']} {dist['version']} in {dist['site']} differs from the frozen source list (sha256 "
                              f"{files_digest(files)[:12]} vs {dist['sha256'][:12]}); refusing to build")  # fmt: skip
        for (rel, text), (_, data) in zip(files, raw):
            digest = sha256_hex(text.encode("utf-8"))
            if PREFIX + rel in paths or digest in digests:
                continue
            paths.add(PREFIX + rel)
            digests.add(digest)
            out.append(Source(PREFIX + rel, text, dist["name"], data))
    return out


def ordered_sources(sources: list[Source], seed: int) -> list[Source]:
    """The order the level takes sources in: .py files, then the rest, each shuffled by `seed`."""
    rng = random.Random(f"corpus-ext:{seed}")
    py = [s for s in sources if s.path.endswith(".py")]
    other = [s for s in sources if not s.path.endswith(".py")]
    rng.shuffle(py)
    rng.shuffle(other)
    return py + other


@dataclass
class BaseInput:
    """What an extension takes from its base: the manifest (and its file's bytes, b"" if not
    read), each file's bytes and text, and the perturbation records' bytes (perturbations.jsonl,
    b"" if none)."""

    manifest: dict
    manifest_raw: bytes
    raw: dict[str, bytes]
    texts: dict[str, str]
    records_raw: bytes


def read_base(base: dict, base_dir: str | Path) -> BaseInput:
    """The base corpus's files from its cache, each checked against its manifest sha256, and its
    records from `base_dir` (runs/corpus/<id>)."""
    raw, texts = {}, {}
    for f in base["files"]:
        raw[f["path"]] = (Path(base["cache_dir"]) / "files" / f["path"]).read_bytes()
        texts[f["path"]] = text_of(raw[f["path"]])
        if texts[f["path"]] is None or sha256_hex(texts[f["path"]].encode("utf-8")) != f["sha256"]:
            raise CorpusError(f"base file {f['path']} does not match its manifest sha256")
    manifest, records = Path(base_dir) / "manifest.json", Path(base_dir) / "perturbations.jsonl"
    return BaseInput(base, manifest.read_bytes() if manifest.exists() else b"", raw, texts, records.read_bytes() if records.exists() else b"")


def _estimator(manifest: dict):
    """The base corpus's estimator, from the calibration file whose sha256 it recorded."""
    payload = manifest["estimator"]
    if payload.get("estimator") != "calib-v1":
        from ..tokens import BYTES4_V0

        if payload == BYTES4_V0.payload():
            return BYTES4_V0
        raise CorpusError(f"the base corpus's estimator {payload.get('estimator')} cannot be rebuilt")
    from ..tokens import CALIBRATION_DIR

    for path in sorted(Path(CALIBRATION_DIR).glob(f"{payload['model']}-[0-9]*.json")):
        estimator = calibration_estimator(json.loads(path.read_text(encoding="utf-8")))
        if estimator.payload() == payload:
            return estimator
    raise CorpusError(f"no calibration file matches the base corpus's estimator {payload['calibration_sha256'][:12]}")


def _reached(tokens: int, window: int, above: int) -> int:
    """The level `tokens` reach at `window`: the largest power of ten, if that is above `above`
    (the base's top level), else tokens // window."""
    rho = 1
    while rho * 10 * window <= tokens:
        rho *= 10
    return rho if rho > above else tokens // window


def extension_plan(base: dict, candidates: list[Source], *, rho: int, seed: int, spec: dict, estimator) -> dict:
    """Which candidates the level takes (whole files, in order, until W x rho tokens, W the base's
    window, counted with the base's estimator), the level reached, the build inputs and the id."""
    window = base["window"]
    base_tokens = base["totals"][str(max(base["rhos"]))]
    target = window * rho
    total, members, tokens_of = base_tokens, [], {}
    for source in candidates:
        if total >= target:
            break
        tokens_of[source.path] = estimator.count(source.text, source_class(source.path))
        total += tokens_of[source.path]
        members.append(source)
    level = rho if total >= target else _reached(total, window, max(base["rhos"]))
    if level <= max(base["rhos"]):
        raise CorpusError(f"the sources add only {total - base_tokens} tokens: no level above C{max(base['rhos'])} is reachable")
    inputs = {"builder": BUILDER_EXT, "base": base["id"], "sources_sha256": sha256_hex(json.dumps(spec, sort_keys=True).encode("utf-8")), "seed": seed,
              "window": window, "rho": level, "estimator": estimator.payload(), "python": spec.get("python")}  # fmt: skip
    if level != rho:  # an unreached level is another build than one that reached it: its own id
        inputs["rho_requested"] = rho
    return {"members": members, "tokens": tokens_of, "base_tokens": base_tokens, "total": total, "target": target, "level": level, "inputs": inputs, "id": corpus_id(inputs)}


def extend_corpus(
    base: dict,
    *,
    rho: int = 10000,
    seed: int = 1,
    spec: dict | None = None,
    runs_dir: str | Path = RUNS_CORPUS,
    cache_dir: str | Path = CACHE_DIR,
    estimator=None,
    base_dir: str | Path | None = None,
    progress: Callable[[str], None] = lambda m: None,
    snapshot: tuple[BaseInput, list[Source]] | None = None,
) -> dict:
    """Build the extension of corpus `base` (its manifest; `base_dir` its runs/corpus/<id>
    directory, default under `runs_dir`) to level `rho` and return the new manifest (module
    docstring). With `snapshot` (eval/snapshot.read_snapshot: the base and the level's files, in
    order), nothing is read from the base's cache or any site-packages, and the snapshot's files
    must be exactly the ones the level takes."""
    started = time.monotonic()
    spec = spec if spec is not None else json.loads(SOURCES_FILE.read_text(encoding="utf-8"))
    estimator = estimator or _estimator(base)
    if snapshot is None:
        given = read_base(base, base_dir if base_dir is not None else Path(runs_dir) / base["id"])
        candidates = ordered_sources(load_sources(spec), seed)
        progress(f"read {len(candidates)} extension files from {len(spec['distributions'])} distributions (digests verified) in {time.monotonic() - started:.1f}s")
    else:
        given, candidates = snapshot
    plan = extension_plan(base, candidates, rho=rho, seed=seed, spec=spec, estimator=estimator)
    if snapshot is not None and len(plan["members"]) != len(candidates):
        raise CorpusError(f"the snapshot holds {len(candidates)} extension files but level C{plan['level']} takes {len(plan['members'])}: not this build's snapshot")
    window, level, total, base_tokens, inputs, cid = base["window"], plan["level"], plan["total"], plan["base_tokens"], plan["inputs"], plan["id"]
    texts = dict(given.texts)
    members = [s.path for s in plan["members"]]
    dist_of = {s.path: s.dist for s in plan["members"]}
    source_sha = {s.path: sha256_hex(s.text.encode("utf-8")) for s in plan["members"]}
    texts.update((s.path, s.text) for s in plan["members"])
    edits_all = []
    for path in members:
        if path.endswith(".py"):
            tree = ast.parse(texts[path])
            edits = literal_edits(path, texts[path], tree, base["seed"])
            if edits:
                new = apply_edits(texts[path], edits)
                if _parses(new):
                    texts[path] = new
                    edits_all += [e[3] for e in sorted(edits, key=lambda e: e[0])]
    progress(f"extension: {len(members)} files, {total - base_tokens} tokens; perturbed {len({e['path'] for e in edits_all})} files with {len(edits_all)} edits")
    out_dir, cache = Path(runs_dir) / cid, Path(cache_dir) / cid
    files_dir = cache / "files"
    for f in base["files"]:
        target_path = files_dir / f["path"]
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(given.raw[f["path"]])
    for path in members:
        target_path = files_dir / path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(texts[path], encoding="utf-8")
    full = [f["path"] for f in base["files"]] + members
    cuts = {**{int(r): int(n) for r, n in base["counts"].items()}, level: len(full)}
    build_index(cache, texts, full, cuts, progress)
    rhos = sorted(cuts)
    manifest = {
        **{k: v for k, v in base.items() if k not in ("id", "files", "cache_dir", "rhos", "targets", "totals", "counts")},
        "id": cid,
        "base": base["id"],
        "builder": BUILDER_EXT,
        "rhos": rhos,
        "targets": {**base["targets"], str(level): window * level},
        "totals": {**base["totals"], str(level): total},
        "counts": {**base["counts"], str(level): len(full)},
        "extension": {**inputs, "rho_requested": rho, "reached": total >= plan["target"], "files": len(members), "tokens": total - base_tokens,
                      "distributions": sorted(set(dist_of.values())), "prefix": PREFIX, "perturbed_edits": len(edits_all)},  # fmt: skip
        "files": base["files"] + [{"path": p, "bytes": len(texts[p].encode("utf-8")), "tokens": plan["tokens"][p], "sha256": sha256_hex(texts[p].encode("utf-8")),
                                   "source_sha256": source_sha[p], "distribution": dist_of[p], "rho": level} for p in members],  # fmt: skip
        "cache_dir": str(cache),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    with open(out_dir / "perturbations.jsonl", "w", encoding="utf-8") as f:
        f.write(text_of(given.records_raw))
        for edit in edits_all:
            f.write(json.dumps(edit, sort_keys=True, ensure_ascii=False) + "\n")
    progress(f"corpus {cid}: C{level} {total} tokens in {len(full)} files ({time.monotonic() - started:.1f}s)")
    return manifest


__all__ = ["BUILDER_EXT", "PREFIX", "SOURCES_FILE", "BaseInput", "Source", "eligible_files", "eligible_raw", "extend_corpus", "extension_plan", "freeze_sources",
           "license_of", "load_sources", "ordered_sources", "permissive", "read_base", "text_of"]

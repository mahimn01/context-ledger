"""The corpus source snapshot (BUILD-V4B decision 6): `corpus extend` never reads a live
site-packages or the base corpus's cache, only this snapshot.

`snapshot_extension` reads what an extension uses, once, and copies it into a content-addressed
store, `data/corpus-src/objects/<sha256[:2]>/<sha256>` (the file's bytes; gitignored): the base
corpus's manifest.json and perturbations.jsonl, every base file, and every source file the
extension level takes, in the level's order (eval/extend.extension_plan over the frozen source
list, its digests verified). It writes the committed manifest
`data/corpus-src/MANIFEST-<corpus id>.tsv`, one row per file in corpus order: the relative corpus
path (manifest.json and perturbations.jsonl for the base's records), the sha256 and size of the
stored bytes, and the source: `base-record:<base id>`, `base:<base id>`, or
`<distribution>==<version>@<site-packages root>` for an extension file. If that manifest exists
(committed), every file read must match its row, or the snapshot is refused; if all its objects
are already in the store and verify, nothing is read at all.

`read_snapshot` gives eval/extend.extend_corpus the base and the level's files from the store
only; a missing object or a hash that differs refuses (CorpusError). The extension built from it
is the one the manifest is named after, or extend refuses.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

from ..ids import sha256_hex
from .corpus import ROOT, CorpusError
from .extend import SOURCES_FILE, BaseInput, Source, _estimator, extension_plan, load_sources, ordered_sources, read_base, text_of

STORE = ROOT / "data" / "corpus-src"
HEADER = "path\tsha256\tbytes\tsource\n"


def object_path(store: str | Path, sha: str) -> Path:
    return Path(store) / "objects" / sha[:2] / sha


def put(store: str | Path, data: bytes) -> bool:
    """Store `data` under its sha256; True if it was new. An existing object must verify."""
    path = object_path(store, sha256_hex(data))
    if path.exists():
        if sha256_hex(path.read_bytes()) != path.name:
            raise CorpusError(f"snapshot object {path} does not match its sha256")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return True


def get(store: str | Path, sha: str, size: int | None = None) -> bytes:
    path = object_path(store, sha)
    if not path.exists():
        raise CorpusError(f"snapshot object {sha[:12]} is missing from {Path(store) / 'objects'}; run `corpus snapshot` (or copy the objects) first")
    data = path.read_bytes()
    if sha256_hex(data) != sha or (size is not None and len(data) != size):
        raise CorpusError(f"snapshot object {sha[:12]} does not match its sha256 or size")
    return data


def site_label(site: str) -> str:
    """A site-packages directory's short label: homebrew-python3.11, else its python directory."""
    parts = Path(site).parts
    return ("homebrew-" if "homebrew" in parts else "") + Path(site).parent.name


def read_manifest(path: str | Path) -> list[tuple[str, str, int, str]]:
    lines = Path(path).read_text(encoding="utf-8").splitlines(keepends=True)
    if not lines or lines[0] != HEADER:
        raise CorpusError(f"{path} is not a corpus-src manifest")
    rows = []
    for line in lines[1:]:
        p, sha, size, source = line.rstrip("\n").split("\t")
        rows.append((p, sha, int(size), source))
    return rows


def render_manifest(rows: list[tuple[str, str, int, str]]) -> str:
    return HEADER + "".join(f"{p}\t{sha}\t{size}\t{source}\n" for p, sha, size, source in rows)


def find_snapshot(store: str | Path, base_id: str) -> Path:
    """The one snapshot manifest in `store` whose base is `base_id`."""
    found = [p for p in sorted(Path(store).glob("MANIFEST-*.tsv")) if f"\tbase-record:{base_id}\n" in p.read_text(encoding="utf-8")]
    if len(found) != 1:
        raise CorpusError(f"{len(found)} snapshot manifests of base {base_id} in {store} (need exactly one; run `corpus snapshot` or pass --snapshot)")
    return found[0]


def complete(store: str | Path, path: str | Path) -> bool:
    """Every object of the manifest is in the store and verifies."""
    try:
        for _, sha, size, _ in read_manifest(path):
            get(store, sha, size)
    except CorpusError:
        return False
    return True


def snapshot_rows(given: BaseInput, members: list[Source], spec: dict) -> list[tuple[str, str, int, str, bytes]]:
    """(path, sha256, size, source, bytes) of every file an extension uses, in corpus order."""
    base_id = given.manifest["id"]
    if not given.manifest_raw:
        raise CorpusError(f"the base's manifest.json was not read: snapshot needs the base's runs/corpus/{base_id} directory")
    records = [("manifest.json", given.manifest_raw)] + ([("perturbations.jsonl", given.records_raw)] if given.records_raw else [])
    version = {d["name"]: (d["version"], site_label(d["site"])) for d in spec["distributions"]}
    rows = [(p, data, f"base-record:{base_id}") for p, data in records]
    rows += [(f["path"], given.raw[f["path"]], f"base:{base_id}") for f in given.manifest["files"]]
    rows += [(s.path, s.raw, f"{s.dist}=={version[s.dist][0]}@{version[s.dist][1]}") for s in members]
    return [(p, sha256_hex(data), len(data), source, data) for p, data, source in rows]


def snapshot_extension(base: dict, base_dir: str | Path, *, rho: int, seed: int, spec: dict | None = None, store: str | Path = STORE, estimator=None,
                       progress: Callable[[str], None] = lambda m: None) -> dict:  # fmt: skip
    """Snapshot what extending `base` to `rho` uses (module docstring) and return a summary:
    the corpus id, the manifest path, files, bytes and new objects."""
    spec = spec if spec is not None else json.loads(SOURCES_FILE.read_text(encoding="utf-8"))
    given = read_base(base, base_dir)
    plan = extension_plan(base, ordered_sources(load_sources(spec), seed), rho=rho, seed=seed, spec=spec, estimator=estimator or _estimator(base))
    rows = snapshot_rows(given, plan["members"], spec)
    path = Path(store) / f"MANIFEST-{plan['id']}.tsv"
    text = render_manifest([r[:4] for r in rows])
    if path.exists() and path.read_text(encoding="utf-8") != text:
        committed = {(r[0], r[3]): r[1] for r in read_manifest(path)}
        differ = [r[0] for r in rows if committed.get((r[0], r[3])) != r[1]]
        raise CorpusError(f"what was read differs from the committed {path.name} ({len(differ)} files, e.g. {differ[:3]}); refusing to snapshot")
    new = sum(put(store, r[4]) for r in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    progress(f"snapshot {plan['id']}: {len(rows)} files, {sum(r[2] for r in rows)} bytes, {new} new objects")
    return {"id": plan["id"], "manifest": path, "files": len(rows), "bytes": sum(r[2] for r in rows), "new_objects": new, "level": plan["level"]}


def read_snapshot(store: str | Path, path: str | Path) -> tuple[BaseInput, list[Source]]:
    """The base and the level's files of a snapshot manifest, from the store only (module
    docstring)."""
    rows = read_manifest(path)
    records = {p: get(store, sha, size) for p, sha, size, source in rows if source.startswith("base-record:")}
    if "manifest.json" not in records:
        raise CorpusError(f"{path} has no base manifest")
    manifest = json.loads(records["manifest.json"])
    raw = {p: get(store, sha, size) for p, sha, size, source in rows if source == f"base:{manifest['id']}"}
    if list(raw) != [f["path"] for f in manifest["files"]]:
        raise CorpusError(f"{path}'s base files are not base {manifest['id']}'s files in order")
    texts = {p: text_of(data) for p, data in raw.items()}
    for f in manifest["files"]:
        if texts[f["path"]] is None or sha256_hex(texts[f["path"]].encode("utf-8")) != f["sha256"]:
            raise CorpusError(f"base file {f['path']} in the snapshot does not match its manifest sha256")
    members = []
    for p, sha, size, source in rows:
        if not source.startswith(("base:", "base-record:")):
            data = get(store, sha, size)
            members.append(Source(p, text_of(data), source.split("==", 1)[0], data))
    given = BaseInput(manifest, records["manifest.json"], raw, texts, records.get("perturbations.jsonl", b""))
    return given, members


def snapshot_id(path: str | Path) -> str:
    return Path(path).name.removeprefix("MANIFEST-").removesuffix(".tsv")


__all__ = ["HEADER", "STORE", "complete", "find_snapshot", "get", "object_path", "put", "read_manifest", "read_snapshot", "snapshot_extension", "snapshot_id", "site_label"]

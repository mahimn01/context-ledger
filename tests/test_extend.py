"""BUILD-V4 item 8: the nested extension level. Built from a fake site-packages directory (never
the machine's): only permissively licensed distributions, verified against the frozen digests;
the base levels byte-identical (files, chunks, BM25 ranking); extension files only at the new
level, perturbed like the base; questions read from the base when the extension has none; and the
source snapshot (BUILD-V4B decision 6) the CLI extends from, never a site-packages or the base's
cache."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from context_ledger.eval import cli
from context_ledger.eval.corpus import CorpusError, CorpusIndex, load_manifest
from context_ledger.eval.extend import PREFIX, extend_corpus, freeze_sources, license_of, permissive
from context_ledger.eval.runner import load_run_questions
from context_ledger.eval.snapshot import object_path, read_manifest, read_snapshot, snapshot_extension
from context_ledger.ids import sha256_hex
from pilot_fixture import v2_fixture_files, write_fixture_tarball

MIX = "anchors=5,probes=3,multihop=2,unanswerable=2,notes=4,supersessions=1,session=3,session_unanswerable=1"


def fake_site(root: Path, name: str, license_line: str, files: dict[str, str]) -> None:
    dist = root / f"{name}-1.0.dist-info"
    dist.mkdir(parents=True)
    (dist / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n{license_line}\n")
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    (dist / "RECORD").write_text("".join(f"{rel},sha256=x,1\n" for rel in files) + f"{dist.name}/METADATA,,\n")


def module(n: int) -> str:
    return f'"""Distractor {n}."""\n\nTIMEOUT_{n} = {100 + n}\n\n\ndef helper_{n}(x, limit={7 + n}):\n    return x + limit\n' * 20


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    work = tmp_path_factory.mktemp("extend")
    tar, sha = write_fixture_tarball(work, v2_fixture_files())
    e = lambda *argv: cli.main(["--workdir", str(work), *argv])  # noqa: E731
    assert e("corpus", "build", "--seed", "3", "--window", "400", "--rho", "10,20", "--version", "v2", "--estimator", "bytes4-v0",
             "--tarball", str(tar), "--sha256", sha) == 0  # fmt: skip
    assert e("questions", "build", "--version", "v3", "--episodes", "2", "--questions-per-episode", "12", "--mix", MIX, "--probe-gap", "4", "--seed", "2") == 0
    site = work / "site" / "lib" / "python3.15" / "site-packages"
    fake_site(site, "alpha", "License-Expression: MIT", {f"alpha/mod{n}.py": module(n) for n in range(30)} | {"alpha/README.md": "# alpha\n" * 50})
    fake_site(site, "beta", "Classifier: License :: OSI Approved :: BSD License", {f"beta/b{n}.py": module(100 + n) for n in range(30)})
    fake_site(site, "gamma", "License: GPLv3", {"gamma/g.py": module(999)})
    fake_site(site, "broken", "License-Expression: MIT", {"broken/b.py": "def f(:\n"})
    spec = freeze_sources([site], work / "sources.json")
    base = load_manifest(None, work / "runs" / "corpus")
    manifest = extend_corpus(base, rho=50, seed=1, spec=spec, runs_dir=work / "runs" / "corpus", cache_dir=work / ".cache" / "corpus")
    return work, base, spec, manifest, site


def test_only_permissive_distributions_and_parseable_files_are_frozen(built):
    _, _, spec, _, _ = built
    assert [d["name"] for d in spec["distributions"]] == ["alpha", "beta"]  # gamma is GPL; broken has no parseable file
    assert permissive("MIT AND PSF-2.0") and permissive("BSD License") and not permissive("GNU Lesser General Public License v3 (LGPLv3)")
    assert license_of("Metadata-Version: 2.1\nName: x\nLicense-Expression: Apache-2.0\n") == "Apache-2.0"


def test_the_base_levels_are_the_base_byte_for_byte(built):
    work, base, _, manifest, _ = built
    assert manifest["base"] == base["id"] and manifest["rhos"] == [10, 20, 50] and manifest["files"][: len(base["files"])] == base["files"]
    for rho in (10, 20):
        rows = [sqlite3.connect(Path(m["cache_dir"]) / "index.sqlite").execute(f"SELECT path, start, end, body FROM chunks_{rho} ORDER BY rowid").fetchall()
                for m in (base, manifest)]  # fmt: skip
        assert rows[0] == rows[1]
        hits = [[(h.path, h.start) for h in CorpusIndex(m["cache_dir"], rho).search("spool limit retries", 10)] for m in (base, manifest)]
        assert hits[0] == hits[1]
    for f in base["files"]:
        assert sha256_hex((Path(manifest["cache_dir"]) / "files" / f["path"]).read_bytes()) == f["sha256"]


def test_extension_files_sit_only_in_the_new_level_perturbed_like_the_base(built):
    work, base, _, manifest, _ = built
    ext = [f for f in manifest["files"] if f["rho"] == 50]
    assert ext and all(f["path"].startswith(PREFIX) for f in ext) and manifest["totals"]["50"] >= 400 * 50
    assert manifest["extension"]["reached"] and manifest["extension"]["perturbed_edits"] > 0
    perturbed = [f for f in ext if f["sha256"] != f["source_sha256"]]
    assert perturbed  # constants and defaults rewritten by the base's seeded rule
    records = (work / "runs" / "corpus" / manifest["id"] / "perturbations.jsonl").read_text().splitlines()
    base_records = (work / "runs" / "corpus" / base["id"] / "perturbations.jsonl").read_text().splitlines()
    assert records[: len(base_records)] == base_records and len(records) > len(base_records)


def test_a_changed_distribution_is_refused(built, tmp_path):
    work, base, spec, _, site = built
    (site / "beta" / "b0.py").write_text(module(100) + "\nEXTRA = 1\n")
    try:
        with pytest.raises(CorpusError, match="differs from the frozen source list"):
            extend_corpus(base, rho=50, seed=1, spec=spec, runs_dir=tmp_path / "runs", cache_dir=tmp_path / "cache")
    finally:
        (site / "beta" / "b0.py").write_text(module(100))


def test_an_unreachable_level_is_labelled_with_what_it_reaches(built, tmp_path):
    work, base, spec, _, _ = built
    manifest = extend_corpus(base, rho=100000, seed=1, spec=spec, runs_dir=tmp_path / "runs", cache_dir=tmp_path / "cache")
    ext = manifest["extension"]
    assert not ext["reached"] and ext["rho"] < 100000 and manifest["rhos"][-1] == ext["rho"]
    assert manifest["id"] != built[3]["id"] and ext["rho_requested"] == 100000  # never the id of a build that reached its level


def test_runs_on_the_extension_read_the_bases_questions(built):
    work, base, _, manifest, _ = built
    rows = load_run_questions(manifest, work / "runs" / "corpus", "v3")
    assert rows == load_run_questions(base, work / "runs" / "corpus", "v3")
    with pytest.raises(Exception):
        load_run_questions(manifest, work / "runs" / "corpus", "v4")  # neither has v4


def test_questions_v4_on_the_extension_count_confusers_at_every_level(built):
    work, _, _, manifest, _ = built
    e = lambda *argv: cli.main(["--workdir", str(work), *argv])  # noqa: E731
    assert e("questions", "build", "--corpus", manifest["id"], "--version", "v4", "--episodes", "2", "--questions-per-episode", "12", "--mix", MIX,
             "--probe-gap", "4") == 0  # fmt: skip
    path = work / "runs" / "corpus" / manifest["id"] / "questions-v4.jsonl"
    meta, *rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert meta["metadata"]["version"] == "v4" and meta["metadata"]["seed"] == 3  # v4's default seed
    assert all(r["qid"].startswith("v4e") and set(r["confusers"]) == {"10", "20", "50"} for r in rows)


def test_the_cli_builds_the_same_extension_from_its_snapshot_only(built, tmp_path, capsys):
    work, base, _, manifest, site = built
    store = tmp_path / "store"
    e = lambda *argv: cli.main(["--workdir", str(work), *argv])  # noqa: E731
    args = ("--base", base["id"], "--rho", "50", "--sources", str(work / "sources.json"), "--store", str(store))
    assert e("corpus", "snapshot", *args) == 0
    tsv = store / f"MANIFEST-{manifest['id']}.tsv"
    rows = read_manifest(tsv)
    ext = [f for f in manifest["files"] if f["rho"] == 50]
    assert [r[0] for r in rows] == ["manifest.json", "perturbations.jsonl", *(f["path"] for f in manifest["files"])]
    assert {r[3] for r in rows[2 : 2 + len(base["files"])]} == {f"base:{base['id']}"} and all(r[3].startswith(("alpha==1.0@", "beta==1.0@")) for r in rows[-len(ext) :])
    capsys.readouterr()
    # extend reads the snapshot only: the base's cache and the site are moved away while it runs
    out = tmp_path / "out"
    hidden = [(Path(base["cache_dir"]), tmp_path / "hidden-cache"), (site, tmp_path / "hidden-site")]
    for a, b in hidden:
        a.rename(b)
    try:
        assert cli.main(["--workdir", str(out), "corpus", "extend", *args]) == 0
        assert e("corpus", "snapshot", *args) == 0  # complete: reads nothing
        assert "is complete" in capsys.readouterr().out
    finally:
        for a, b in hidden:
            b.rename(a)
    again = load_manifest(manifest["id"], out / "runs" / "corpus")
    drop = lambda m: {k: v for k, v in m.items() if k != "cache_dir"}  # noqa: E731
    assert again["id"] == manifest["id"] and drop(again) == drop(manifest)  # the same build, byte for byte but the cache path
    for name in ("perturbations.jsonl",):
        assert (out / "runs" / "corpus" / manifest["id"] / name).read_bytes() == (work / "runs" / "corpus" / manifest["id"] / name).read_bytes()
    for f in manifest["files"]:
        assert (Path(again["cache_dir"]) / "files" / f["path"]).read_bytes() == (Path(manifest["cache_dir"]) / "files" / f["path"]).read_bytes()
    q = "SELECT path, start, end, body FROM chunks_50 ORDER BY rowid"
    assert sqlite3.connect(Path(again["cache_dir"]) / "index.sqlite").execute(q).fetchall() == sqlite3.connect(Path(manifest["cache_dir"]) / "index.sqlite").execute(q).fetchall()
    # another seed would build another corpus than the snapshot's: refused
    assert cli.main(["--workdir", str(tmp_path / "other"), "corpus", "extend", *args, "--seed", "2"]) == 2


def test_extend_refuses_a_missing_or_altered_object(built, tmp_path):
    work, base, spec, manifest, _ = built
    store = tmp_path / "store"
    s = snapshot_extension(base, work / "runs" / "corpus" / base["id"], rho=50, seed=1, spec=spec, store=store)
    rows = read_manifest(s["manifest"])
    victim = object_path(store, rows[-1][1])
    data = victim.read_bytes()
    victim.write_bytes(data + b"#")
    with pytest.raises(CorpusError, match="does not match"):
        read_snapshot(store, s["manifest"])
    victim.unlink()
    with pytest.raises(CorpusError, match="is missing"):
        read_snapshot(store, s["manifest"])
    victim.write_bytes(data)
    given, members = read_snapshot(store, s["manifest"])
    assert extend_corpus(given.manifest, rho=50, seed=1, spec=spec, runs_dir=tmp_path / "runs", cache_dir=tmp_path / "cache", snapshot=(given, members))["id"] == s["id"]


def test_snapshot_refuses_sources_that_differ_from_the_committed_manifest(built, tmp_path):
    work, base, spec, manifest, site = built
    store = tmp_path / "store"
    s = snapshot_extension(base, work / "runs" / "corpus" / base["id"], rho=50, seed=1, spec=spec, store=store)
    text = s["manifest"].read_text()
    s["manifest"].write_text(text.replace(read_manifest(s["manifest"])[-1][1], "0" * 64))  # as if the committed snapshot had other bytes
    with pytest.raises(CorpusError, match="differs from the committed"):
        snapshot_extension(base, work / "runs" / "corpus" / base["id"], rho=50, seed=1, spec=spec, store=store)

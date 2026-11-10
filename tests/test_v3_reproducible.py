"""BUILD-V4 item 10: v3 stays exactly reproducible. B3, B4 and L (l-v3) run offline with the scripted
fake model on the pilot fixture corpus, under both conditions with questions v3, and every output
they write is compared byte for byte (by sha256) with digests frozen from the tree BEFORE any v4
code existed (commit 5af0069): results.jsonl, calls.jsonl, each B4 transcript, the head hash and
the event count of each L ledger, and run.json's keys and its configuration (corpus_id aside: the
fixture tarball's gzip header holds a timestamp). v4 adds new code paths and labels only; if this
test fails, a v3 output changed."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from context_ledger.eval import cli
from context_ledger.store import Ledger
from pilot_fixture import v2_fixture_files, write_fixture_tarball

SESSION_MIX = "anchors=5,probes=3,multihop=2,unanswerable=2,notes=4,supersessions=1,session=3,session_unanswerable=1"

CONFIG_KEYS = ("live", "arms", "rho", "model", "window", "smoke", "episodes", "questions_per_episode", "estimator", "prompts", "pricing", "questions",
               "questions_version", "questions_sha256", "l_mechanism", "harness_patch", "read_cap_tokens", "condition")  # fmt: skip

# Frozen at 5af0069 (before v4) with tests/test_v3_reproducible.py:v3_digests.
FROZEN: dict[str, str] = {
    "allowed/rho-10/B4/episode-01.jsonl": "2d9982d0b570851764123ec0a3f78725f0132fec72169de737bad75ce8c14a59",
    "allowed/rho-10/B4/episode-02.jsonl": "4f45dd10948b756a0adb0cd574f13c964c58a713c0e49a51b038953ebd345936",
    "allowed/rho-10/L/episode-01.sqlite": "569:36ee7cdc61ef83c6f400c7bbe847ed139425be3106c9c3a4c0c29e395b859efe",
    "allowed/rho-10/L/episode-02.sqlite": "521:d62dd69ed20aa73bfcf205e6793916ac7421bd62d3d65daa855b984de7670ef2",
    "allowed/rho-10/calls.jsonl": "314b792c7cdc438ad22d9fe433c3a4ac7bb08a7544e581377ef242d2418fa705",
    "allowed/rho-10/config": "d5f4e4d25ecd032dc28108bd03b6e968bb974669f905383f342fd5d3474f3ba3",
    "allowed/rho-10/results.jsonl": "4bee1f58b7a00979d86d98eb3973998c17238f29828366dc05d47f54dc643f0d",
    "allowed/rho-10/run.json keys": "arms,budget_usd,condition,config_hash,corpus_id,episodes,estimator,harness_patch,incomplete,key_profile,l_mechanism,leakscan,live,model,pressure,pricing,prompts,questions,questions_per_episode,questions_sha256,questions_version,read_cap_tokens,rho,run_id,smoke,spend,spend_ledger,stopped,tool_definitions,units,window",
    "allowed/rho-20/B4/episode-01.jsonl": "68bc3df17f053f7ec0d3190cadae1d96d63b5410770d3e291bdf46f244f13eb5",
    "allowed/rho-20/B4/episode-02.jsonl": "b838ffcfbf38ecfb540c047b005f3f93083f4a29c57d6027a5fa4f4eccf576e7",
    "allowed/rho-20/L/episode-01.sqlite": "581:a3a49b265f0eacad2e75acff879ef483357b7875fa050a25226b6a81b62eee17",
    "allowed/rho-20/L/episode-02.sqlite": "522:daf5b962581d5056b7460b6640943804b7e66bf916877b94dbbce9f68c9a416a",
    "allowed/rho-20/calls.jsonl": "11c5a38f5bd594ca321a7ae6a2e42f9c0857263261fef90476dcd5c69c788081",
    "allowed/rho-20/config": "6a29ee2307a7529f35c8a3280a530f494b8977502db635feef071786f66cf05d",
    "allowed/rho-20/results.jsonl": "457aae3dc342f5e677824d51908af6d51b653dc68a04fd79ac98d20a52057116",
    "allowed/rho-20/run.json keys": "arms,budget_usd,condition,config_hash,corpus_id,episodes,estimator,harness_patch,incomplete,key_profile,l_mechanism,leakscan,live,model,pressure,pricing,prompts,questions,questions_per_episode,questions_sha256,questions_version,read_cap_tokens,rho,run_id,smoke,spend,spend_ledger,stopped,tool_definitions,units,window",
    "forced/rho-10/B4/episode-01.jsonl": "32e3b0d07cca1fdb16dd385fd253d6cab01e1848564b29242933c0599410fb4b",
    "forced/rho-10/B4/episode-02.jsonl": "ba99617caa1feb2391a08253c3d5d68fbbf1259dbac6b5a2a0dc89092c2d92fa",
    "forced/rho-10/L/episode-01.sqlite": "714:0a0b28bafb5ac11a69a856e7d99e4467a4f6b2d3b9b08a0dad024ba48f181661",
    "forced/rho-10/L/episode-02.sqlite": "635:94fb426e3ea218def0f95fe91a9363d2297c3da37e23b72874fa48e0bd750a94",
    "forced/rho-10/calls.jsonl": "14bf939f3072cab106d4195c7f4eeff44c66d2dde5fa40e6b6965583be41d27b",
    "forced/rho-10/config": "4ebba2b38e133c89f90d3e294cbeca21be877ee1dda8d8aafb81be1a1036a187",
    "forced/rho-10/results.jsonl": "9cad7d23e0cc135cce1543ede32c419cf97f6d06249b3b4536e3f42b7345d61a",
    "forced/rho-10/run.json keys": "arms,budget_usd,condition,config_hash,corpus_id,episodes,estimator,harness_patch,incomplete,key_profile,l_mechanism,leakscan,live,model,pressure,pricing,prompts,questions,questions_per_episode,questions_sha256,questions_version,read_cap_tokens,rho,run_id,smoke,spend,spend_ledger,stopped,tool_definitions,units,window",
    "forced/rho-20/B4/episode-01.jsonl": "7a61d84c2f27724df210bea7f86ed784f47603cabd16f5e97ad261b832b418df",
    "forced/rho-20/B4/episode-02.jsonl": "500ea79775a9305f97784579546617ead19e1401e4782d28337e83fabdf3eb3f",
    "forced/rho-20/L/episode-01.sqlite": "677:798e7751ab2f43ee425eac3ef27f58a2bf47b65aa7d656e28d4d29318c67e427",
    "forced/rho-20/L/episode-02.sqlite": "638:036034b9bc98e6eb089ef003f78c587e224f0fce935884c4eb847c1e2cf2edb7",
    "forced/rho-20/calls.jsonl": "cedd4dc65784f7f35ba5433d3eed0550c9b48f8f4e161239952465c0cbad5395",
    "forced/rho-20/config": "23c90d1b3f4d63996fdd7b419f57ec53bc5684da676091f19598b0f74818e3f0",
    "forced/rho-20/results.jsonl": "c973db2f92e13f40c4a90a6e2dbdc8d6dd6125733dc30f2003ebb15134feb545",
    "forced/rho-20/run.json keys": "arms,budget_usd,condition,config_hash,corpus_id,episodes,estimator,harness_patch,incomplete,key_profile,l_mechanism,leakscan,live,model,pressure,pricing,prompts,questions,questions_per_episode,questions_sha256,questions_version,read_cap_tokens,rho,run_id,smoke,spend,spend_ledger,stopped,tool_definitions,units,window",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def v3_digests(work: Path) -> dict[str, str]:
    """Build the fixture corpus and questions v3 in `work`, run B3,B4,L in both conditions, and
    return {artifact: sha256 or value}."""
    tar, sha = write_fixture_tarball(work, v2_fixture_files())
    e = lambda *argv: cli.main(["--workdir", str(work), *argv])  # noqa: E731
    assert e("corpus", "build", "--seed", "3", "--window", "400", "--rho", "10,20", "--version", "v2", "--estimator", "bytes4-v0",
             "--tarball", str(tar), "--sha256", sha) == 0  # fmt: skip
    assert e("questions", "build", "--version", "v3", "--episodes", "2", "--questions-per-episode", "12", "--mix", SESSION_MIX, "--probe-gap", "4") == 0
    out: dict[str, str] = {}
    for condition in ("allowed", "forced"):
        run_id = f"repro-{condition}"
        assert e("run", "--arms", "B3,B4,L", "--questions", "v3", "--rho", "10,20", "--episodes", "1-2", "--window", "3000", "--budget", "2",
                 "--run-id", run_id, "--condition", condition) == 0  # fmt: skip
        root = work / "runs" / run_id
        for part in sorted(root.glob("rho-*")):
            key = f"{condition}/{part.name}"
            out[f"{key}/results.jsonl"] = _sha(part / "results.jsonl")
            out[f"{key}/calls.jsonl"] = _sha(part / "calls.jsonl")
            run = json.loads((part / "run.json").read_text())
            out[f"{key}/run.json keys"] = ",".join(sorted(run))
            # the configuration but corpus_id: the fixture tarball's gzip header holds a timestamp
            out[f"{key}/config"] = hashlib.sha256(json.dumps({k: run[k] for k in CONFIG_KEYS}, sort_keys=True).encode()).hexdigest()
            for transcript in sorted((part / "B4").glob("episode-*.jsonl")):
                out[f"{key}/B4/{transcript.name}"] = _sha(transcript)
            for db in sorted((part / "L").glob("episode-*.sqlite")):
                ledger = Ledger(db)
                try:
                    out[f"{key}/L/{db.name}"] = f"{len(ledger.events())}:{ledger.head()}"
                finally:
                    ledger.close()
    return out


@pytest.mark.skipif(not FROZEN, reason="digests not frozen yet")
def test_b3_b4_and_l_v3_outputs_are_byte_identical_to_the_pre_v4_tree(tmp_path):
    got = v3_digests(tmp_path)
    assert got == FROZEN

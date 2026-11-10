from __future__ import annotations

import importlib.util
import json
import shutil
import sqlite3
from pathlib import Path

import pytest

from context_ledger import cli
from context_ledger.store import Ledger

ROOT = Path(__file__).resolve().parents[1]


def load_demo():
    spec = importlib.util.spec_from_file_location("run_demo", ROOT / "demo" / "run_demo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def demo():
    return load_demo()


@pytest.fixture(scope="module")
def kept_db(demo, tmp_path_factory):
    db = tmp_path_factory.mktemp("demo") / "demo.sqlite"
    result = demo.main(db_path=db)
    assert result["ok"]
    return db


def test_demo_main(demo, capsys):
    result = demo.main()
    assert result["ok"] is True
    assert result["violations"] == []
    assert {"P7", "P11"} <= set(result["tampered_props"])
    assert len(result["refetches"]) >= 1
    assert result["verdicts"] == {"Q1": "accept", "Q2": "accept-after-refetch", "Q3": "abstain"}
    assert f"{demo.MODULE_PATH}#DEFAULT_TIMEOUT" in result["stale_spans"]
    trace = capsys.readouterr().out
    assert "backstop fold" in trace and "ok: True" in trace


def test_demo_is_deterministic(demo, capsys):
    assert demo.main() == demo.main()


def test_cli_check_clean_ledger(kept_db, capsys):
    assert cli.main(["check", str(kept_db)]) == 0
    assert "0 violations" in capsys.readouterr().out


def test_cli_replay_and_export(kept_db, capsys):
    with Ledger(kept_db) as ledger:
        events = ledger.events()
    call_id = next(e.payload["call_id"] for e in reversed(events) if e.type == "call")
    recorded = next(e.payload for e in events if e.type == "call" and e.payload["call_id"] == call_id)
    capsys.readouterr()
    assert cli.main(["replay", str(kept_db), "--call", call_id]) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["manifest_hash"] == recorded["manifest_hash"] and body["read_map"] == recorded["read_map"]
    assert body["manifest"][0][1] == "system"
    assert cli.main(["export-prov", str(kept_db)]) == 0
    prov = json.loads(capsys.readouterr().out)
    assert f"clp:call/{call_id}" in prov["activity"]
    assert any(k.startswith("clp:span/") for k in prov["entity"]) and any(k.startswith("clp:blob/") for k in prov["entity"])
    assert prov["used"] and prov["wasDerivedFrom"] and prov["wasGeneratedBy"]
    claims = [r for r in prov["wasGeneratedBy"].values() if r["prov:entity"].startswith("clp:claim/")]
    assert claims and all(r["prov:activity"] in prov["activity"] for r in claims)


def test_cli_errors(kept_db, tmp_path, capsys):
    assert cli.main(["check", str(tmp_path / "missing.sqlite")]) == 2
    assert cli.main(["replay", str(kept_db), "--call", "call_missing"]) == 2


def test_cli_check_reports_violations(kept_db, tmp_path, capsys):
    db = tmp_path / "copy.sqlite"
    shutil.copy(kept_db, db)
    for suffix in ("-wal", "-shm"):
        side = Path(str(kept_db) + suffix)
        if side.exists():
            shutil.copy(side, Path(str(db) + suffix))
    conn = sqlite3.connect(db)
    conn.execute("DROP TRIGGER events_no_update")  # simulate tampering outside the library
    conn.execute("UPDATE events SET type = 'tampered' WHERE seq = 5")
    conn.commit()
    conn.close()
    capsys.readouterr()
    assert cli.main(["check", str(db)]) == 1
    assert "P11" in capsys.readouterr().out

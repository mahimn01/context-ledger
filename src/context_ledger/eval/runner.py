"""The pilot runner: one run of chosen arms over chosen questions at one rho and model.

Layout of runs/<run_id>/:
    run.json          the configuration, prompt and price-table hashes, corpus id, estimator,
                      whether the run completed, why it stopped, the leakscan result, each arm's
                      tool-definition tokens and the `pressure` manipulation check
    results.jsonl     one QuestionResult per arm and question
    calls.jsonl       one row per model call: usage, exact cost, predicted prompt and cache read
    B4/episode-NN.jsonl, L/episode-NN.sqlite    the agent transcripts and the ledgers
    fake-spend.jsonl  the offline run's spend ledger (a live run uses .clp-spend/ instead)

Without --live the client is the FakeAnthropicServer (emulating the 2026-10-31 models) with the
ScriptedPilotModel, through the same GuardedClient and SpendGuard as a live run.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path

from ..anthropic.fake import FakeAnthropicServer
from ..ids import canonical_json, sha256_hex
from ..live.guarded import GuardedClient
from ..live.keys import DEFAULT_PROFILE
from ..live.spend import KEY_LEDGERS, SpendGuard
from ..pricing import PricingTable, usd
from ..tokens import decimal_str
from ..index import INDEX_FILE, INDEX_VERSION, AddressIndex
from .address_tools import MAX_SEARCHES, preresolve_cap
from .arms import ARMS, CONDITIONS, MAX_CALLS_EPISODE, HARNESS_PATCH, HARNESS_PATCH_V4FIX, L_MECHANISM, ArmConfig, CallLog, RunStopped, set_aside_episode
from .arms_v4 import ARMS_V4, B5_MECHANISM, L4_MECHANISM, L4_NOGATE_MECHANISM
from .corpus import CorpusIndex
from .fakemodel import ScriptedPilotModel
from .prompts import COMPACTION_PROMPT_VERSION, COMPACTION_SHA256, FORCED_PROMPTS, PILOT_PROMPT_VERSION, prompt_arms, prompt_sha256
from .questions import smoke_questions
from .stats import nearest_rank
from .tools import read_cap

ALL_ARMS = {**ARMS, **ARMS_V4}  # BUILD-V4: B5 and L4 beside B3, B4 and L (BUILD-V4B: and L4-nogate)
V4_ARMS = tuple(ARMS_V4)
PROMPT_ARM = {"L4-nogate": "L4"}  # an arm that sends another arm's prompt (L4-nogate: L4's, BUILD-V4B decision 3)
MAX_REPAIRS_V4 = 2  # anthropic/client_v4.MAX_REPAIRS, recorded in run.json

def episode_list(rows: list[dict], episodes: int | str | list[int]) -> list[int]:
    """The episodes a run covers: the first N of the questions file (an int or "N"), the
    inclusive range "a-b", or an explicit list; ValueError if one is not in the file."""
    present = sorted({r["episode"] for r in rows})
    if isinstance(episodes, str) and "-" in episodes:
        a, b = (int(x) for x in episodes.split("-", 1))
        wanted = list(range(a, b + 1))
    elif isinstance(episodes, (int, str)):
        wanted = present[: int(episodes)]
    else:
        wanted = list(episodes)
    missing = [e for e in wanted if e not in present]
    if missing or not wanted:
        raise ValueError(f"episodes {missing or episodes} are not in the questions file (it has {present[0]}-{present[-1]})" if present else "no questions")
    return wanted


def select_questions(rows: list[dict], *, episodes: int | str | list[int], per_episode: int | None, smoke: bool) -> dict[str, list[dict]]:
    """The questions each arm is asked. Smoke: the first 3 of episode 1 (one unanswerable), and
    for B4 and L also the fold probe of the first answerable one, as a 4th question. Otherwise
    the episodes `episode_list` names, each with its first `per_episode` questions (None: all
    of them, as many as the questions file holds per episode)."""
    if smoke:
        three, probe = smoke_questions(rows)
        with_probe = three + ([probe] if probe is not None else [])
        return {"B3": three, "B4": with_probe, "L": with_probe, "B5": three, "L4": with_probe, "L4-nogate": with_probe}
    chosen = []
    for e in episode_list(rows, episodes):
        chosen += sorted((r for r in rows if r["episode"] == e), key=lambda r: r["pos"])[:per_episode]
    return {arm: chosen for arm in ALL_ARMS}


def questions_sha256(chosen: dict[str, list[dict]]) -> str:
    """The hash of every question row a run asks (gold, notes and tags included), by qid."""
    rows = {q["qid"]: q for qs in chosen.values() for q in qs}
    return sha256_hex(json.dumps([rows[k] for k in sorted(rows)], sort_keys=True, ensure_ascii=False).encode("utf-8"))


def question_tags(q: dict, rho: int, by_qid: dict[str, dict]) -> dict:
    """What a results row copies from its question (the arms never see it): template, hops,
    cross_file, fold_probe, probe_gap and the confusers at the run's rho (v2 questions; v1
    questions have hops 1, or 0 when unanswerable, no cross-file hop and no confuser count)."""
    gap = q.get("probe_gap")
    if gap is None and q.get("probe_of") in by_qid:
        gap = q["pos"] - by_qid[q["probe_of"]]["pos"]
    return {"template": q["template"], "hops": q.get("hops", 1 if q["answerable"] else 0), "cross_file": bool(q.get("cross_file", False)),
            "fold_probe": bool(q["fold_probe"]), "probe_gap": gap, "confusers": (q.get("confusers") or {}).get(str(rho))}  # fmt: skip


def offline_client(run_dir: Path, *, run_id: str, pricing: PricingTable, budget_usd: str, global_cap_usd: str | int | None = None, responder=None,
                   key_profile: str = DEFAULT_PROFILE, use_reserve: bool = False):  # fmt: skip
    """The fake server and scripted model (or `responder`) behind a GuardedClient with the run's
    own spend ledger, under the caps of `key_profile` (as a live run with that key would be)."""
    guard = SpendGuard(run_dir / "fake-spend.jsonl", lock_path=run_dir / "fake-spend.lock", run_id=run_id, pricing=pricing,
                       key=KEY_LEDGERS[key_profile], use_reserve=use_reserve, run_cap_usd=budget_usd, global_cap_usd=global_cap_usd)  # fmt: skip
    server = FakeAnthropicServer(responder=responder or ScriptedPilotModel(), pricing=pricing, emulate=True)
    return GuardedClient(server.as_fake_client(), guard, sleep=lambda s: None), guard, server


class RunConfigMismatch(Exception):
    """A run directory already holds a run with another configuration: refuse to mix them."""


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """Replace `path` with `rows` atomically (a temporary file, fsync, then os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def _append(path: Path, rows: list[dict]) -> None:
    """Add `rows` to the end of `path`, atomically."""
    _write_jsonl(path, _read_jsonl(path) + rows)


def _split(path: Path, keep) -> None:
    """Keep the rows of `path` for which keep(row) holds; append the others, unchanged, to the
    sibling <name>.discarded.jsonl (nothing is ever deleted)."""
    rows = _read_jsonl(path)
    gone = [r for r in rows if not keep(r)]
    if gone:
        _append(path.with_name(path.stem + ".discarded.jsonl"), gone)
        _write_jsonl(path, [r for r in rows if keep(r)])


def completed_units(run_dir: Path) -> set[tuple[str, int]]:
    """(arm, episode) units whose results were written as a completed episode."""
    return {(r["arm"], r["episode"]) for r in _read_jsonl(run_dir / "results.jsonl") if not r.get("partial")}


def run_pilot(
    *,
    manifest: dict,
    rows: list[dict],
    arms: list[str],
    rho: int,
    model: str,
    window: int,
    episodes: int | str | list[int],
    per_episode: int | None,
    smoke: bool,
    run_dir: Path,
    run_id: str,
    pricing: PricingTable,
    estimator,
    client: GuardedClient,
    live: bool,
    budget_usd: str,
    prompt_version: str = PILOT_PROMPT_VERSION,
    progress: Callable[[str], None] | None = None,
    tool_definitions: dict[str, int] | None = None,
    questions_version: str = "v1",
    condition: str = "allowed",
    key_profile: str = DEFAULT_PROFILE,
    max_calls_episode: int = MAX_CALLS_EPISODE,
) -> dict:
    """Run the arms unit by unit, a unit being one (arm, model, rho, episode), and write
    run.json, results.jsonl and calls.jsonl. Idempotent and resumable:

    - run.json is written first with the hash of the run's configuration (everything that
      shapes the results; not the budget). A directory holding another configuration is refused
      (RunConfigMismatch).
    - Each completed episode's results are written atomically. On a restart with the same
      configuration, completed units are skipped; a unit cut short is redone, after its rows in
      results.jsonl, calls.jsonl and L/folds.jsonl move to *.discarded.jsonl and its L ledger
      or B4 transcript is set aside (its spend stays in the spend ledger; nothing is deleted).
    - A guard refusal or halt stops the run cleanly: the partial episode's results are written
      flagged "partial" (analysable as far as they went; redone on a resume) and run.json says
      incomplete. A killed process leaves an open reservation that `spend resolve
      --assume-spent` closes before the next start.
    `progress` gets one line per completed episode (counts and cost only).

    `tool_definitions` are each arm's tool-definition tokens (from the envelope file); if None
    they are measured with the client's free count_tokens (with minus without `tools`) before
    the arms run, since L's backstop measures W as count_tokens minus L's. They and the
    `pressure` block (the manipulation check, `pressure`) are written to run.json.

    `condition` (BUILD-V3): "allowed" or "forced", recorded in run.json and every results row;
    the forced condition needs prompt clp-v3-forced, and that prompt needs the forced condition.
    `harness_patch` (BUILD-V3FIX, arms.HARNESS_PATCH) is recorded in run.json and every L row
    (B3 and B4 do not use the patched code); a run without it ran unpatched, and since it is
    part of the configuration such a run is never resumed patched (RunConfigMismatch: rerun
    into a new run id). BUILD-V4FIX: a run with a v4 arm records HARNESS_PATCH +
    "+" + arms.HARNESS_PATCH_V4FIX instead (a run of B3, B4 and L alone keeps v3's).

    `key_profile` (BUILD-KEY) names the API key that paid: written to run.json and every results
    and calls row, but not part of the configuration (like the budget), so a run started with
    one key may be resumed with the other and each row says which key paid for it.

    BUILD-V4: arms B5 and L4 need a clp-v4 prompt (it has their tool paragraphs) and the corpus's
    address index (.cache/corpus/<id>/address-v4.sqlite, `eval index build`). Only a run with one
    of them adds to the configuration: a "v4" block (the mechanisms, the index version and the
    Merkle root of C-rho, the repair and search bounds) and, when not 90, max_calls_episode. A run
    of B3, B4 and L alone writes exactly v3's configuration."""
    unknown = [a for a in arms if a not in ALL_ARMS]
    if unknown:
        raise ValueError(f"unknown arms {unknown}; choose from {sorted(ALL_ARMS)}")
    if condition not in CONDITIONS or (condition == "forced") != (prompt_version in FORCED_PROMPTS):
        raise ValueError(f"condition {condition!r} does not go with prompt {prompt_version!r}: forced goes with {' or '.join(FORCED_PROMPTS)} and only with them")
    unprompted = [a for a in arms if PROMPT_ARM.get(a, a) not in prompt_arms(prompt_version)]
    if unprompted:
        raise ValueError(f"prompt {prompt_version} has no tool paragraph for {unprompted}: B5 and L4 need clp-v4-allowed or clp-v4-forced")
    v4 = [a for a in arms if a in V4_ARMS]
    address = None
    patch = f"{HARNESS_PATCH}+{HARNESS_PATCH_V4FIX}" if v4 else HARNESS_PATCH  # BUILD-V4FIX (arms.HARNESS_PATCH_V4FIX)
    if v4:
        address = AddressIndex(Path(manifest["cache_dir"]) / INDEX_FILE, rho)  # IndexError_ if it is missing or lacks rho
    run_dir.mkdir(parents=True, exist_ok=True)
    chosen = select_questions(rows, episodes=episodes, per_episode=per_episode, smoke=smoke)
    config = {
        "live": live, "arms": arms, "rho": rho, "model": model, "window": window, "smoke": smoke, "episodes": episodes,
        "questions_per_episode": per_episode, "corpus_id": manifest["id"], "estimator": estimator.payload(),
        "prompts": {"version": prompt_version, "sha256": {arm: prompt_sha256(arm, prompt_version) for arm in prompt_arms(prompt_version)},
                    "compaction": {"version": COMPACTION_PROMPT_VERSION, "sha256": COMPACTION_SHA256}},
        "pricing": {"version": pricing.version, "pricing_hash": pricing.pricing_hash(), "verified": pricing.verified},
        "questions": {arm: [q["qid"] for q in chosen[arm]] for arm in arms},
        "questions_version": questions_version,
        "questions_sha256": questions_sha256(chosen),
        "l_mechanism": L_MECHANISM,
        "harness_patch": patch,
        "read_cap_tokens": read_cap(window),
        "condition": condition,
    }  # fmt: skip
    if v4:
        config["v4"] = {"l4_mechanism": L4_MECHANISM, "b5_mechanism": B5_MECHANISM, "max_repairs": MAX_REPAIRS_V4, "max_searches": MAX_SEARCHES,
                        "preresolve_cap_tokens": preresolve_cap(window),
                        "index": {"version": INDEX_VERSION, "rho": rho, "root": address.root, "symbols": address.level["symbols"], "files": address.level["files"]},
                        **({"l4_nogate_mechanism": L4_NOGATE_MECHANISM} if "L4-nogate" in arms else {})}  # fmt: skip
    if max_calls_episode != MAX_CALLS_EPISODE:
        config["max_calls_episode"] = max_calls_episode
    config_hash = sha256_hex(canonical_json(config))
    run_json = run_dir / "run.json"
    if run_json.exists():
        prior = json.loads(run_json.read_text(encoding="utf-8")).get("config_hash")
        if prior != config_hash:
            raise RunConfigMismatch(f"{run_dir} holds a run with config hash {prior}, not {config_hash}; use a new --run-id")
    else:
        run_json.write_text(json.dumps({"run_id": run_id, **config, "config_hash": config_hash, "key_profile": key_profile, "status": "running"},
                                       indent=1, sort_keys=True) + "\n")  # fmt: skip
    done = completed_units(run_dir)
    _split(run_dir / "results.jsonl", lambda r: not r.get("partial"))
    _split(run_dir / "calls.jsonl", lambda r: (r["arm"], r["episode"]) in done)
    _split(run_dir / "L" / "folds.jsonl", lambda r: ("L", r["episode"]) in done)
    for arm in ("L4", "L4-nogate"):
        _split(run_dir / arm / "folds.jsonl", lambda r, arm=arm: (arm, r["episode"]) in done)
    units = [(arm, e) for arm in arms for e in sorted({q["episode"] for q in chosen[arm]})]
    by_qid = {q["qid"]: q for q in rows}
    tag = lambda r: {**r.to_dict(), "rho": rho, "model": model, "key_profile": key_profile, **question_tags(by_qid[r.qid], rho, by_qid),  # noqa: E731
                     **({"harness_patch": patch} if r.arm in ("L", "L4", "L4-nogate") else {})}  # B3, B4 and B5 do not go through the patched code  # fmt: skip
    if tool_definitions is None:  # before the arms: L's backstop measures W minus them
        tool_definitions, source = measure_tool_definitions(client, model), "measured with count_tokens"
    else:
        source = "envelope calibration file"
    missing = [a for a in v4 if a not in tool_definitions]
    if missing:  # the envelope file predates the v4 arms: their definitions are counted now (free)
        tool_definitions = {**tool_definitions, **measure_tool_definitions(client, model, v4_tools(missing))}
        source += f"; {', '.join(missing)} measured with count_tokens"
    index = CorpusIndex(manifest["cache_dir"], rho)
    cfg = ArmConfig(model=model, window=window, pricing=pricing, estimator=estimator, run_dir=run_dir, prompt_version=prompt_version,
                    tool_definitions=tool_definitions, condition=condition, address=address, max_calls_episode=max_calls_episode)  # fmt: skip
    log = CallLog(run_dir / "calls.jsonl", key_profile=key_profile)
    stopped = None
    try:
        for arm, episode in units:
            if (arm, episode) in done:
                continue
            set_aside_episode(run_dir, arm, episode)
            questions = [q for q in chosen[arm] if q["episode"] == episode]
            first_call = len(log.rows)
            try:
                results = ALL_ARMS[arm](questions, index, client, cfg, log)
            except RunStopped as stop:
                partial = [{**tag(r), "partial": True} for r in getattr(stop, "partial", [])]
                _append(run_dir / "results.jsonl", partial)
                stopped = {"reason": stop.reason, "detail": stop.detail, "arm": arm, "episode": episode}
                break
            _append(run_dir / "results.jsonl", [tag(r) for r in results])
            done.add((arm, episode))
            if progress is not None:
                calls = log.rows[first_call:]
                progress(f"progress: run {run_id} rho {rho} {model} {arm} episode {episode}: {len(results)} questions, {len(calls)} calls, "
                         f"{usd(sum(c['cost_pusd'] for c in calls))}; {len(done)}/{len(units)} units done")  # fmt: skip
    finally:
        index.close()
        if address is not None:
            address.close()
    summary = {
        "run_id": run_id,
        **config,
        "config_hash": config_hash,
        "budget_usd": budget_usd,
        "key_profile": key_profile,
        "tool_definitions": {"tokens": {arm: tool_definitions[arm] for arm in arms}, "source": source},
        "pressure": pressure(_read_jsonl(run_dir / "calls.jsonl"), _read_jsonl(run_dir / "results.jsonl"), window, tool_definitions),
        "units": {"done": len(done & set(units)), "total": len(units)},
        "incomplete": stopped is not None,
        "stopped": stopped,
        "spend": client.guard.status(),
        "spend_ledger": str(client.guard.path),
    }
    run_json.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def measure_tool_definitions(client, model: str, tools_by_arm: dict[str, list[dict]] | None = None) -> dict[str, int]:
    """Each arm's tool-definition tokens: count_tokens of one user message with the arm's tools
    minus without (free; the fake server in offline runs). `tools_by_arm` defaults to the v3
    arms' (calibrate.arm_tools)."""
    from .calibrate import arm_tools

    def count(**body) -> int:
        result = client.messages.count_tokens(model=model, messages=[{"role": "user", "content": [{"type": "text", "text": "x"}]}], **body)
        return result["input_tokens"] if isinstance(result, dict) else result.input_tokens

    bare = count()
    return {arm: (count(tools=tools) - bare if tools else 0) for arm, tools in (tools_by_arm if tools_by_arm is not None else arm_tools()).items()}


def v4_tools(arms: list[str]) -> dict[str, list[dict]]:
    """The tool definitions B5, L4 and L4-nogate send, sorted by name as they send them."""
    from ..tools import CTX_TOOLS
    from .address_tools import ADDRESS_TOOLS, B5_TOOLS

    ordered = lambda tools: sorted((json.loads(json.dumps(t)) for t in tools), key=lambda t: t["name"])  # noqa: E731
    return {arm: ordered({"B5": B5_TOOLS, "L4": CTX_TOOLS + ADDRESS_TOOLS}[PROMPT_ARM.get(arm, arm)]) for arm in arms}


def pressure(calls: list[dict], results: list[dict], window: int, tool_definitions: dict[str, int]) -> dict:
    """The manipulation check, per arm: the p50, p95 (nearest rank) and max of the actual prompt
    (input + cache read + cache write, as the API reported) minus the arm's tool definitions,
    over W, on answer calls, and how many of them were above W; folds and compactions per
    episode; and the share of answer calls made with the window at least 50% full (by that same
    actual measure)."""
    out = {}
    for arm in sorted({r["arm"] for r in results} | {c["arm"] for c in calls}):
        answer = [c for c in calls if c["arm"] == arm and c.get("kind", "answer") == "answer"]
        actual = [c["usage"]["input_tokens"] + c["usage"]["cache_read_input_tokens"] + c["usage"]["cache_creation_input_tokens"] - tool_definitions.get(arm, 0)
                  for c in answer]  # fmt: skip
        ratios = [Fraction(a, window) for a in actual]
        mine = [r for r in results if r["arm"] == arm]
        episodes = len({r["episode"] for r in mine}) or 1
        show = lambda v: None if v is None else decimal_str(v, 4)  # noqa: E731
        out[arm] = {
            "answer_calls": len(answer),
            "prompt_over_w_p50": show(nearest_rank(ratios, 50) if ratios else None),
            "prompt_over_w_p95": show(nearest_rank(ratios, 95) if ratios else None),
            "prompt_over_w_max": show(max(ratios) if ratios else None),
            "calls_over_w": sum(a > window for a in actual),
            "share_calls_half_full": show(Fraction(sum(2 * a >= window for a in actual), len(actual)) if actual else None),
            "folds_per_episode": show(Fraction(sum(r.get("folds", 0) for r in mine), episodes)),
            "compactions_per_episode": show(Fraction(sum(r.get("compactions", 0) for r in mine), episodes)),
        }
    return out


def load_run_questions(manifest: dict, runs_dir: Path, version: str | None) -> list[dict]:
    """The questions file a run used: v1's questions.jsonl, v2's questions-v2.jsonl, v2s's
    questions-v2s.jsonl (v2 with session facts, BUILD-V2c), v3's questions-v3.jsonl (BUILD-V3) or
    v4's questions-v4.jsonl (BUILD-V4). An extension corpus (BUILD-V4, eval/extend.py) without
    that version's file reads its base corpus's: its levels up to the base's are the base's."""
    from .questions import QuestionError, load_questions
    from .questions_v2 import load_questions_v2

    if version not in ("v2", "v2s", "v3", "v4"):
        return load_questions(manifest, runs_dir)
    try:
        return load_questions_v2(manifest, runs_dir, version)
    except QuestionError:
        if not manifest.get("base") or not (Path(runs_dir) / manifest["base"]).is_dir():
            raise
        return load_questions_v2({"id": manifest["base"]}, runs_dir, version)


def load_run(run_dir: Path) -> tuple[dict, list[dict], list[dict]]:
    """(run.json, results rows, call rows) of a run directory."""
    read = lambda name: [json.loads(line) for line in (run_dir / name).read_text(encoding="utf-8").splitlines()] if (run_dir / name).exists() else []  # noqa: E731
    return json.loads((run_dir / "run.json").read_text(encoding="utf-8")), read("results.jsonl"), read("calls.jsonl")

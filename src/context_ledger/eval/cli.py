"""python -m context_ledger.eval <command>: the pilot harness from the command line.

    corpus build --seed S [--estimator auto|bytes4-v0|calib-v1] [--rho 10,100,1000] [--window W] [--version v1|v2]
    corpus snapshot --base ID [--rho 10000] [--seed 1] [--sources FILE] [--store DIR]
                                               BUILD-V4B: copy what the extension uses into data/corpus-src
                                               (content-addressed) and write its MANIFEST-<id>.tsv
    corpus extend --base ID [--rho 10000] [--seed 1] [--sources FILE] [--store DIR] [--snapshot FILE]
                                               BUILD-V4: the nested extension level (eval/extend.py), read
                                               only from the snapshot
    index build [--corpus ID]                  BUILD-V4: the address index (address-v4.sqlite) of a corpus
    questions build [--corpus ID] [--seed S] [--version v1|v2|v2s|v3|v4] [--episodes N]
    questions overlap [--corpus ID] [--version v4]   BUILD-V4B: the file's items against questions v3 by episode group
                    v2: [--questions-per-episode Q] [--mix anchors=..,probes=..,multihop=..,unanswerable=..]
                    v2s: as v2, the mix also with notes=..,supersessions=..,session=..,session_unanswerable=..
                    v3: the v2s generator under its own label and file (questions-v3.jsonl)
                    v4: the same under the label v4 (questions-v4.jsonl), seed 3 by default, fresh: nothing overlaps
                        questions v3 episodes 1-12 or 19 (extras from the rho-10 core if needed; questions_v2)
                        [--probe-gap G] [--dry-run]   (--dry-run prints capacity, writes nothing)
    calibrate [--live] [--counter fake]        offline needs --counter fake
    calibrate --envelope [--live] [--counter fake|count_tokens] --model M [--out DIR]
                                               fit the envelope (calibration/<model>-envelope-<date>.json)
    plan [--measured RUN] [--window W]        with --measured, first the run's measured cost per
                                               episode-arm unit (every rho part, its condition)
    plan --v4 [--measured PILOT_RUN...] [--measured-v3 RUN...] [--max-calls-episode 260]
              [--model M] [--episodes N] [--rhos 10,100,...] [--arms ...] [--conditions allowed,forced]
                                               BUILD-V4B: the v4 pilot's and grids' cost at 260 calls per
                                               episode: from the pilot runs where measured, else the v3 grid
                                               (runs/v3-grid-*); per-run --budget suggestions
    run [--live] --arms B3,B4,L[,B5,L4,L4-nogate] --rho 10[,100,...] --episodes N|a-b [--questions-per-episode K] --model M
        --budget USD [--window W] [--smoke] [--grid] [--questions v1|v2|v2s|v3|v4] [--condition allowed|forced]
        [--max-calls-episode N]
                                               the condition picks the prompt (clp-v3-allowed or
                                               clp-v3-forced; clp-v4-* when B5, L4 or L4-nogate runs); a
                                               conflicting --prompt is refused
        calibrate, plan and run also take [--key research|aperture] [--use-reserve]: the API key
        (default research), which picks its own spend ledger and caps together with it; plan
        budgets against that ledger. --use-reserve releases the key's reserve (research: $20 of
        its $200) and is recorded in its ledger.
    score RUN
    report RUN
    analyze RUN... --out DIR [--resamples N]   the pre-registered tests (PREREG sections 4-5), plus
                                               by_template.csv and pressure.csv (labelled)
    analyze-v3 RUN... --out DIR [--resamples N]   the PREREG-v3 analysis (eval/analyze_v3.py): cells.csv,
                                               tests.csv, tests.json and figure_data.csv, plus
                                               main_figure.png if matplotlib is importable
    analyze-v4 RUN... --out DIR [--resamples N]   the v4 analysis (eval/analyze_v4.py): six confirmatory tests per
                                               model (L4 vs B3 non-inferiority, confident errors vs B3 and B4,
                                               TOST of accuracy over rho in both conditions, forced unlabelled
                                               fabrications vs B4), Holm within model; the rest exploratory;
                                               cells, tests and figure_data.csv
    quick RUN                                  pilot table per arm x rho (official grader)
    diagnose-abstain RUN                       why L abstained on answerable questions
    spend status|verify|resolve [--assume-spent] [--key research|aperture] [--spend-file F]
                                               status: every key's ledger and the combined total;
                                               verify and resolve: the --key's ledger (default research)
    leakscan

Nothing here makes a network call without --live, and every --live command goes through
`open_live` (the live gate, a verified price table, the SpendGuard, then the key). Without
--live, `run` uses the FakeAnthropicServer through the same GuardedClient and SpendGuard (under
the --key's caps), with its spend ledger at runs/<run_id>/fake-spend.jsonl. `--workdir` moves
runs/, .cache/corpus/ and .clp-spend/ (for tests); the default is the repo.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path

from ..anthropic.request import PILOT_THINKING
from ..envelope import OFFLINE_ENVELOPE, EnvelopeError, EnvelopeMissing, load_envelope
from ..live.entry import open_live
from ..live.gate import LiveRefused
from ..live.keys import DEFAULT_PROFILE, PROFILES, EnvironmentKeyProvider, KeyUnavailable
from ..live.redact import forget_secrets, safe_message
from ..live.spend import (
    KEY_LEDGERS,
    BudgetRefused,
    OpenReservation,
    SpendError,
    SpendGuard,
    ledger_status,
    read_spend,
    render_combined,
    render_status,
    totals,
)
from ..pricing import PICO_PER_USD, PricingError, PricingTable, usd
from ..tokens import BYTES4_V0, CalibEstimator, CalibrationMissing, estimator_or_fallback, load_calibration, window_estimator
from . import corpus as corpus_mod
from . import questions_v2
from .analyze import analyze, load_runs
from .analyze_v3 import analyze_v3, load as load_v3
from .analyze_v4 import analyze_v4, load as load_v4
from .calibrate import (
    CalibrationError,
    CountTokensCounter,
    CountTokensRequestCounter,
    FakeEnvelopeCounter,
    FakeTokenCounter,
    calibrate,
    calibrate_envelope,
    draw_sample,
    render_envelope_report,
    render_report,
    write_calibration,
    write_envelope,
)
from .corpus import CorpusError, build_corpus, corpus_texts, load_manifest, read_sources
from .extend import SOURCES_FILE, extend_corpus
from .snapshot import complete, find_snapshot, read_snapshot, snapshot_extension, snapshot_id
from ..index import INDEX_FILE, IndexError_, build_address_index
from .leakscan import scan
from .plan_v4 import ARMS as V4_PLAN_ARMS, DEFAULT_MAX_CALLS as V4_MAX_CALLS, plan_v4, render_plan_v4
from .plan import HAIKU, SAFETY as PLAN_SAFETY, SONNET, SONNET_RHOS, PlanError, episode_costs, latest_live_smoke, measured_params, plan, render_episode_costs, render_plan
from .prompts import CONDITION_PROMPTS, CONDITION_PROMPTS_V4, FORCED_PROMPTS, PROMPTS
from .questions import QuestionError, build_questions, load_question_metadata, load_questions
from .report import build_report
from .diagnose import diagnose_abstain, quick, render_diagnose, render_quick
from .runner import V4_ARMS, RunConfigMismatch, episode_list, load_run_questions, offline_client, run_pilot
from .score import score, score_run

ROOT = corpus_mod.ROOT


def _out(text: str) -> None:
    sys.stdout.write(text + "\n")


def _err(text: str) -> None:
    sys.stderr.write(text + "\n")


def _dirs(args) -> dict[str, Path]:
    base = Path(args.workdir) if args.workdir else ROOT
    return {"runs": base / "runs", "corpus_runs": base / "runs" / "corpus", "cache": base / ".cache" / "corpus", "spend": base / ".clp-spend", "base": base,
            "store": Path(args.store) if getattr(args, "store", None) else base / "data" / "corpus-src"}


def _estimator(name: str, model: str):
    if name == "bytes4-v0":
        return BYTES4_V0, None
    if name == "calib-v1":
        return load_calibration(model), None
    return estimator_or_fallback(model)


def cmd_corpus(args) -> int:
    d = _dirs(args)
    if args.seed is None:
        _err("error: corpus build needs --seed")
        return 2
    args.rho = args.rho or "10,100,1000"
    estimator, warning = _estimator(args.estimator, args.model)
    if warning:
        _err(warning)
    kwargs = {"tarball": args.tarball, "sha256": args.sha256} if args.tarball else {}
    manifest = build_corpus(seed=args.seed, estimator=estimator, estimator_warning=warning, window=args.window, version=args.version,
                            rhos=[int(r) for r in args.rho.split(",")], runs_dir=d["corpus_runs"], cache_dir=d["cache"], **kwargs)  # fmt: skip
    _out(f"corpus {manifest['id']}: " + ", ".join(f"C{r} {manifest['totals'][r]} tokens in {manifest['counts'][r]} files" for r in manifest["totals"]))
    return 0


def cmd_extend(args) -> int:
    """`corpus extend`: the nested extension level of a base corpus (eval/extend.py), read only
    from its source snapshot (eval/snapshot.py): never a site-packages, never the base's cache."""
    d = _dirs(args)
    path = Path(args.snapshot) if args.snapshot else find_snapshot(d["store"], args.base)
    given, members = read_snapshot(d["store"], path)
    base = given.manifest
    if args.base and args.base != base["id"]:
        _err(f"error: snapshot {path.name} extends base {base['id']}, not {args.base}")
        return 2
    spec = json.loads(Path(args.sources).read_text(encoding="utf-8")) if args.sources else None
    manifest = extend_corpus(base, rho=int(args.rho or 10000), seed=1 if args.seed is None else args.seed, spec=spec, runs_dir=d["corpus_runs"].resolve(),
                             cache_dir=d["cache"].resolve(), progress=_err, snapshot=(given, members))  # fmt: skip
    if manifest["id"] != snapshot_id(path):
        _err(f"error: the build from {path.name} is corpus {manifest['id']}, not {snapshot_id(path)}: different --rho, --seed or --sources")
        return 2
    ext = manifest["extension"]
    level = str(ext["rho"])
    _out(f"corpus {manifest['id']} (base {base['id']}, from snapshot {path.name}): C{level} {manifest['totals'][level]} tokens in {manifest['counts'][level]} files; "
         f"{ext['files']} extension files from {len(ext['distributions'])} distributions add {ext['tokens']} tokens"
         + ("" if ext["reached"] else f"; rho {ext['rho_requested']} was NOT reached, the level is labelled C{level}"))  # fmt: skip
    return 0


def cmd_snapshot(args) -> int:
    """`corpus snapshot`: copy every file `corpus extend` will use into the content-addressed
    store and write (or check) its committed manifest (eval/snapshot.py). Reads nothing if a
    manifest of this base is already complete in the store."""
    d = _dirs(args)
    try:
        existing = find_snapshot(d["store"], args.base)
    except CorpusError:
        existing = None
    if existing is not None and complete(d["store"], existing):
        _out(f"snapshot {snapshot_id(existing)} is complete in {d['store']} ({existing.name}); nothing read")
        return 0
    base_dir = Path(args.base) if Path(args.base).is_dir() else d["corpus_runs"] / args.base
    spec = json.loads(Path(args.sources).read_text(encoding="utf-8")) if args.sources else None
    s = snapshot_extension(load_manifest(base_dir), base_dir, rho=int(args.rho or 10000), seed=1 if args.seed is None else args.seed, spec=spec, store=d["store"],
                           progress=_err)  # fmt: skip
    _out(f"snapshot {s['id']}: {s['files']} files, {s['bytes']} bytes ({s['new_objects']} new objects) in {d['store'] / 'objects'}; manifest {s['manifest']}")
    return 0


def cmd_index(args) -> int:
    """`index build`: the address index of a corpus, in its cache directory (BUILD-V4)."""
    d = _dirs(args)
    manifest = load_manifest(args.corpus, d["corpus_runs"])
    out = Path(manifest["cache_dir"]) / INDEX_FILE
    meta = build_address_index(corpus_texts(manifest), {f["path"]: f["rho"] for f in manifest["files"]}, manifest["rhos"], out, corpus_id=manifest["id"], progress=_err)
    for rho, level in meta["levels"].items():
        _out(f"C{rho}: {level['symbols']} names ({level['names']} without parameters) in {level['files']} files; Merkle root {level['root']}")
    _out(f"wrote {out}")
    return 0


def cmd_questions(args) -> int:
    d = _dirs(args)
    manifest = load_manifest(args.corpus, d["corpus_runs"])
    if args.action == "overlap":
        return _questions_overlap(args, d, manifest)
    if args.seed is None:
        args.seed = questions_v2.V4_SEED if args.version == "v4" else 1
    if args.version in ("v2", "v2s", "v3", "v4"):
        questions = args.questions_per_episode or questions_v2.QUESTIONS
        default = questions_v2.Mix.default_session if args.version in ("v2s", "v3", "v4") else questions_v2.Mix.default
        mix = questions_v2.Mix.parse(args.mix) if args.mix else default(questions)
        if mix.has_session != (args.version in ("v2s", "v3", "v4")):
            _err("error: session keys in --mix (notes, supersessions, session, session_unanswerable) go with --version v2s, v3 or v4, and those need them")
            return 2
        if args.dry_run:
            _out(questions_v2.render_capacity(questions_v2.capacity(manifest, seed=args.seed, questions=questions, mix=mix, probe_gap=args.probe_gap, runs_dir=d["corpus_runs"],
                                                                    version=args.version)))  # fmt: skip
            return 0
        rows = questions_v2.build_questions_v2(manifest, seed=args.seed, episodes=args.episodes, questions=questions, mix=mix, probe_gap=args.probe_gap,
                                               runs_dir=d["corpus_runs"], version=args.version)  # fmt: skip
        m = questions_v2.metadata(rows, manifest, args.seed, mix, questions, args.probe_gap, args.version)
        _out(f"{args.version}: {m['questions']} questions in {m['episodes']} episodes of {questions}: {m['answerable']} answerable ({m['fold_probes']} fold probes, "
             f"{m['multihop']} multi-hop, {m['cross_file']} cross-file), {m['unanswerable']} unanswerable; {m['distinct_gold_files']} gold files; "
             f"{m['with_confusers']} with confusers; templates {m['template_mix']}")  # fmt: skip
        if "session" in m:
            sm = m["session"]
            _out(f"session facts: {sm['notes']} notes, {sm['supersessions']} updates, {sm['probes']} answerable session probes "
                 f"({sm['superseded_probes']} after an update), {sm['unanswerable']} unanswerable; gaps {sm['gaps']}")  # fmt: skip
        if args.version == "v4":
            f = json.loads((d["corpus_runs"] / manifest["id"] / questions_v2.V4_FILENAME).read_text(encoding="utf-8").split("\n", 1)[0])["metadata"]["fresh"]
            _out(f"fresh: {f['episodes_filter_only']} episodes with the overlap filter alone (reference {f['reference']['file']} episodes 1-12 and 19, "
                 f"sha256 {f['reference']['sha256'][:12]}); extras {'used' if f['extras'] else 'not needed'}; {f['episodes_available']} episodes available; "
                 f"kept out {f['excluded']}; corpus items by origin {f['origins']}")  # fmt: skip
        return 0
    if args.dry_run or args.mix or args.questions_per_episode:
        _err("error: --dry-run, --mix and --questions-per-episode are for --version v2, v2s or v3")
        return 2
    build_questions(manifest, seed=args.seed, episodes=args.episodes, runs_dir=d["corpus_runs"])
    m = load_question_metadata(manifest, d["corpus_runs"])
    _out(f"{m['questions']} questions in {m['episodes']} episodes: {m['answerable']} answerable ({m['fold_probes']} fold probes), "
         f"{m['unanswerable']} unanswerable; {m['distinct_gold_files']} gold files ({len(m['shared_gold_files'])} shared by two episodes); "
         f"templates {m['template_mix']}")  # fmt: skip
    return 0


def _questions_overlap(args, d, manifest) -> int:
    """`questions overlap`: a questions file against questions v3 by episode group (questions_v2.
    overlap_report); exit 1 if any item overlaps the reference."""
    version = args.version if args.version != "v1" else "v4"
    rows = questions_v2.load_questions_v2(manifest, d["corpus_runs"], version)
    reference = questions_v2.reference_path(manifest, d["corpus_runs"])
    report = questions_v2.overlap_report(rows, questions_v2.question_rows(reference.read_bytes()))
    _out(questions_v2.render_overlap(report, f"questions {version} of {manifest['id']} vs {reference.parent.name}/{reference.name}"))
    return 1 if report["overlapping_reference"] else 0


def cmd_calibrate(args) -> int:
    d = _dirs(args)
    if not args.live and args.counter != "fake":
        _err("error: offline calibration needs --counter fake (the real count_tokens counter needs --live)")
        return 2
    if args.envelope:
        return _calibrate_envelope(args, d)
    kwargs = {"path": args.tarball, "sha256": args.sha256} if args.tarball else {}
    files = read_sources(**kwargs)
    sample = draw_sample(files, args.seed)
    if args.live:
        pricing = PricingTable.verified_default()
        ctx = open_live(args, os.environ, keys=EnvironmentKeyProvider(args.key), pricing=pricing, run_id=f"calibrate-{_stamp()}",
                        run_cap_usd="0", use_reserve=args.use_reserve, spend_dir=d["spend"])  # count_tokens is free: no spend at all  # fmt: skip
        try:
            result = calibrate(sample, CountTokensCounter(ctx.client, args.model), model=args.model, seed=args.seed)
            path = write_calibration(result, args.out)
        finally:
            leak = scan(root=ROOT, secret=ctx.secret)
            ctx.close()
            forget_secrets()
        _out(render_report(result))
        _out(f"wrote {path}")
        _out(leak.render())
        return 0 if leak.ok else 1
    result = calibrate(sample, FakeTokenCounter(), model=args.model, seed=args.seed)
    path = write_calibration(result, args.out or d["runs"] / "calibration-fake")
    _out(render_report(result))
    _out(f"wrote {path} (fake counter: never used for live or pilot paths)")
    return 0


def _calibrate_envelope(args, d) -> int:
    """Fit the envelope against count_tokens (live; free) or the fake counter (offline), with
    the content measured by the calib-v2 slopes of the model's calib-v1 calibration."""
    content = window_estimator(load_calibration(args.model))
    thinking = PILOT_THINKING.get(args.model)
    if args.live:
        pricing = PricingTable.verified_default()
        ctx = open_live(args, os.environ, keys=EnvironmentKeyProvider(args.key), pricing=pricing, run_id=f"calibrate-envelope-{_stamp()}",
                        run_cap_usd="0", use_reserve=args.use_reserve, spend_dir=d["spend"])  # count_tokens is free: no spend at all  # fmt: skip
        try:
            result = calibrate_envelope(CountTokensRequestCounter(ctx.client, args.model), content, model=args.model, thinking=thinking)
            path = write_envelope(result, args.out)
        finally:
            leak = scan(root=ROOT, secret=ctx.secret)
            ctx.close()
            forget_secrets()
        _out(render_envelope_report(result))
        _out(f"wrote {path}")
        _out(leak.render())
        return 0 if leak.ok else 1
    result = calibrate_envelope(FakeEnvelopeCounter(OFFLINE_ENVELOPE, content), content, model=args.model, thinking=thinking)
    path = write_envelope(result, args.out or d["runs"] / "calibration-fake")
    _out(render_envelope_report(result))
    _out(f"wrote {path} (fake counter: never used for live or pilot paths)")
    return 0


def run_estimator(model: str, live: bool, calibration=None, directory=None):
    """(window estimator, tool definitions per arm or None, warnings) for a run. With a calib-v1
    calibration and an envelope file fitted against it: calib-v3. A live run refuses (raises
    EnvelopeMissing) without one; offline, a missing envelope falls back to the offline default
    envelope with a warning (tool definitions are then measured on the fake server)."""
    warnings = []
    if calibration is None:
        calibration, warning = _estimator("auto", model)
        warnings += [warning] if warning else []
    if not isinstance(calibration, CalibEstimator):
        if live:
            raise EnvelopeMissing(f"a live run needs a calib-v1 calibration and an envelope for {model}")
        return window_estimator(calibration), None, warnings
    try:
        fitted = load_envelope(model, directory)
    except EnvelopeMissing as exc:
        if live:
            raise
        warnings.append(f"warning: {exc}; offline, using the default envelope {OFFLINE_ENVELOPE.to_dict()}")
        return window_estimator(calibration, OFFLINE_ENVELOPE), None, warnings
    if fitted.calibration_sha256 != calibration.calibration_sha256:
        message = f"the envelope for {model} was fitted against calibration {fitted.calibration_sha256[:12]}, not the one in force ({calibration.calibration_sha256[:12]})"
        if live:
            raise EnvelopeMissing(message + "; re-run `calibrate --live --envelope`")
        warnings.append("warning: " + message)
    return window_estimator(calibration, fitted.envelope, fitted.sha256), fitted.tool_definitions, warnings


def _remaining(d, profile: str, use_reserve: bool = False) -> int:
    """What the key's own ledger has left under its cap (less its reserve without --use-reserve)."""
    key = KEY_LEDGERS[profile]
    t = totals(read_spend(key.path(d["spend"])))
    return key.cap_pusd(use_reserve) - t.spent - t.outstanding


def _plan_params(d, window: int, measured: str | None):
    """Measured params from the smoke run named (or the latest live smoke run at `window`), or
    None for the prior. Raises PlanError if a named run has another window."""
    smoke = Path(measured) if measured else latest_live_smoke(d["runs"], window)
    if smoke is None:
        return None
    if not (smoke / "run.json").exists():  # a run over several rho: its first part
        parts = sorted(smoke.glob("*/run.json"))
        smoke = parts[0].parent if parts else smoke
    params = measured_params(smoke)
    if params[HAIKU].window != window:
        raise PlanError(f"smoke run {smoke} ran at W = {params[HAIKU].window}, not {window}: its costs do not transfer")
    return params


def cmd_plan(args) -> int:
    d = _dirs(args)
    if args.v4:
        dirs = [Path(p) for p in args.measured_v3] if args.measured_v3 else sorted(p for p in d["runs"].glob("v3-grid-*") if p.is_dir() and not p.name.endswith("-Lrerun"))
        custom = None
        if any(x is not None for x in (args.model, args.episodes, args.rhos, args.arms, args.conditions)):
            custom = {"model": args.model or HAIKU, "episodes": args.episodes or 10, "conditions": tuple((args.conditions or "allowed,forced").split(",")),
                      "rhos": tuple(int(r) for r in (args.rhos or "10,100,1000,10000").split(",")), "arms": tuple((args.arms or ",".join(V4_PLAN_ARMS)).split(","))}  # fmt: skip
        pilot = [p if p.is_dir() else d["runs"] / p for p in map(Path, args.measured or [])]
        _out(render_plan_v4(plan_v4(dirs, pilot, args.max_calls_episode, custom, _remaining(d, args.key, args.use_reserve))))
        return 0
    if args.measured and len(args.measured) > 1:
        _err("error: --measured takes one smoke run without --v4")
        return 2
    measured = args.measured[0] if args.measured else None
    if measured:
        _out(render_episode_costs(episode_costs(Path(measured))))
    pricing = PricingTable.verified_default()
    _out(render_plan(plan(pricing, _plan_params(d, args.window, measured), _remaining(d, args.key, args.use_reserve), window=args.window)))
    return 0


def _stamp() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def cmd_run(args) -> int:
    d = _dirs(args)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    prompt = args.prompt or (CONDITION_PROMPTS_V4 if any(a in V4_ARMS for a in arms) else CONDITION_PROMPTS)[args.condition]
    if (prompt in FORCED_PROMPTS) != (args.condition == "forced"):
        _err(f"error: --prompt {prompt} conflicts with --condition {args.condition} (forced goes with {' or '.join(FORCED_PROMPTS)}, and only with them)")
        return 2
    pricing = PricingTable.verified_default()
    calibration, warning = _estimator("auto", args.model)  # the SpendGuard's conservative fallback uses it
    if warning:
        _err(warning)
    # every arm measures W with it: calib-v3 (calib-v2 content plus the envelope); a live run
    # refuses without a fitted envelope file
    estimator, tool_definitions, warnings = run_estimator(args.model, bool(args.live), calibration)
    for w in warnings:
        _err(w)
    manifest = load_manifest(args.corpus, d["corpus_runs"])
    rows = load_run_questions(manifest, d["corpus_runs"], args.questions)
    grid_rhos = list(SONNET_RHOS) if args.model == "claude-sonnet-5-5" else list(corpus_mod.RHOS)
    rhos = [int(r) for r in args.rho.split(",")] if args.rho else grid_rhos if args.grid else [10]
    try:
        episode_list(rows, args.episodes)
        a, _, b = str(args.episodes).partition("-")
        n_episodes = int(b) - int(a) + 1 if b else int(a)  # as requested: what the budget check plans for
    except ValueError as exc:
        _err(f"error: {exc}")
        return 2
    run_id = args.run_id or f"{'grid' if args.grid else 'smoke' if args.smoke else 'run'}-{args.model.removeprefix('claude-')}-{args.condition}-{_stamp()}"
    run_dir = d["runs"] / run_id
    if args.grid:
        item = "haiku_grid" if args.model == "claude-haiku-5-5" else "sonnet_subset"
        remaining = _remaining(d, args.key, args.use_reserve)
        p = plan(pricing, _plan_params(d, args.window, None), remaining, window=args.window)
        expected = p["per_episode"][item] * n_episodes
        if expected * PLAN_SAFETY > min(remaining, int(Fraction(str(args.budget)) * PICO_PER_USD)):
            _err(f"error: the grid's estimate {usd(expected)} x 1.2 is more than --budget or the {usd(remaining)} the {args.key} key's SpendGuard has left")
            return 2
    if args.use_reserve:
        _err(f"--use-reserve: the {args.key} key's reserve is released for run {run_id} (recorded in its spend ledger)")
    ctx = None
    if args.live:
        ctx = open_live(args, os.environ, keys=EnvironmentKeyProvider(args.key), pricing=pricing, run_id=run_id, run_cap_usd=args.budget,
                        use_reserve=args.use_reserve, spend_dir=d["spend"], estimator=calibration)  # fmt: skip
        client, guard = ctx.client, ctx.guard
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        client, guard, _ = offline_client(run_dir, run_id=run_id, pricing=pricing, budget_usd=args.budget, key_profile=args.key, use_reserve=args.use_reserve)
    try:
        summaries = []
        for rho in rhos:
            target = run_dir / f"rho-{rho}" if args.grid or len(rhos) > 1 else run_dir
            summaries.append(run_pilot(manifest=manifest, rows=rows, arms=arms, rho=rho, model=args.model, window=args.window,
                                       episodes=args.episodes, per_episode=args.questions_per_episode, smoke=args.smoke, run_dir=target,
                                       run_id=run_id, pricing=pricing, estimator=estimator, client=client, live=bool(args.live),
                                       budget_usd=args.budget, prompt_version=prompt, progress=_out,
                                       tool_definitions=tool_definitions, questions_version=args.questions, condition=args.condition,
                                       key_profile=args.key, max_calls_episode=args.max_calls_episode))  # fmt: skip
            if summaries[-1]["incomplete"]:
                break
    finally:
        # A live run scans every runtime artifact, the tracked tree and the history, and looks
        # for the exact key while it is still in memory; an offline run scans its own directory.
        leak = scan(root=ROOT, secret=ctx.secret) if ctx is not None else scan(root=ROOT, runtime=[run_dir], git=False)
        guard.close()
        if ctx is not None:
            forget_secrets()
    for target in sorted({run_dir, *run_dir.glob("rho-*")}):
        manifest_path = target / "run.json"
        if manifest_path.exists():
            data = json.loads(manifest_path.read_text())
            data["leakscan"] = "pass" if leak.ok else "fail"
            manifest_path.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    records = read_spend(guard.path)
    for s in summaries:
        status = f"INCOMPLETE ({s['stopped']['reason']})" if s["incomplete"] else "complete"
        _out(spend_line(run_id, s["rho"], status, records, s["spend"]["run_cap_pusd"], args.key))
    _out(f"leakscan: {'pass' if leak.ok else 'FAIL'}; results in {run_dir}")
    return 0 if leak.ok else 1


def spend_line(run_id: str, rho: int, status: str, records: list[dict], run_cap_pusd: int, key_profile: str) -> str:
    """The end-of-run spend line: what this run spent (the spend-ledger rows with its run_id
    only) against its own cap, and the key's ledger total against that key's hard cap."""
    t = totals(records)
    key = KEY_LEDGERS[key_profile]
    reserve = f", ${key.hard_max_usd - key.reserve_usd} usable without --use-reserve" if key.reserve_usd else ""
    return (f"run {run_id} rho {rho}: {status}; this run spent {usd(t.per_run.get(run_id, 0))} of its {usd(run_cap_pusd)} run cap "
            f"({key_profile} key ledger: {usd(t.spent)} of the ${key.hard_max_usd} hard cap{reserve})")  # fmt: skip


def _run_dir(args) -> Path:
    path = Path(args.run)
    return path if path.is_dir() else _dirs(args)["runs"] / args.run


def cmd_score(args) -> int:
    d = _dirs(args)
    run_dir = _run_dir(args)
    found = [p for p in (run_dir / "run.json", *sorted(run_dir.glob("*/run.json"))) if p.exists()]
    if not found:
        _err(f"error: no run in {run_dir}")
        return 2
    run = json.loads(found[0].read_text())
    manifest = load_manifest(run["corpus_id"], d["corpus_runs"])
    rows = score_run(run_dir, load_run_questions(manifest, d["corpus_runs"], run.get("questions_version")), corpus_texts(manifest))
    outcomes = {}
    for r in rows:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
    _out(f"scored {len(rows)} answers: " + ", ".join(f"{k} {v}" for k, v in sorted(outcomes.items())))
    _out(f"hallucinations: unsupported {sum(r['hallucination'] == 'unsupported' for r in rows)}, misread {sum(r['hallucination'] == 'misread' for r in rows)}, caught {sum(r['caught'] for r in rows)}")
    return 0


def cmd_report(args) -> int:
    run_dir = _run_dir(args)
    build_report(run_dir, resamples=args.resamples)
    _out((run_dir / "report.md").read_text())
    _out(f"wrote {run_dir / 'report.md'} and {run_dir / 'report.json'}")
    return 0


def cmd_analyze(args) -> int:
    d = _dirs(args)
    corpora: dict[str, tuple[list[dict], dict[str, str]]] = {}

    def score_fn(run: dict, results: list[dict]) -> list[dict]:
        key = (run["corpus_id"], run.get("questions_version", "v1"))  # gold from the run's own questions file
        if key not in corpora:
            manifest = load_manifest(run["corpus_id"], d["corpus_runs"])
            corpora[key] = (load_run_questions(manifest, d["corpus_runs"], key[1]), corpus_texts(manifest))
        questions, texts = corpora[key]
        return score(results, questions, texts)

    run_dirs = [p if p.is_dir() else d["runs"] / p for p in map(Path, args.runs)]
    scores, calls, folds, superseded = load_runs(run_dirs, score_fn)
    if not scores:
        _err(f"error: no results in {', '.join(map(str, run_dirs))}")
        return 2
    out = Path(args.out)
    result = analyze(scores, calls, folds, out, resamples=args.resamples, superseded=superseded, meta={"runs": [str(p) for p in run_dirs]})
    for t in result["tests"]:
        shown = lambda v: "n/a" if v is None else (f"{float(v):.4f}")  # noqa: E731
        _out(f"{t['test']:<22} {t['model']:<18} {t['family']:<9} estimate {shown(t['estimate'])} CI [{shown(t['ci_low'])}, {shown(t['ci_high'])}] "
             f"p {shown(t['p'])} holm {shown(t['p_holm'])}: {t['verdict']}")  # fmt: skip
    _out(f"wrote cells.csv, tests.csv, tests.json, exclusions.csv ({len(result['exclusions'])} rows) and figure_data.csv to {out}")
    return 0


def cmd_analyze_v3(args) -> int:
    run_dirs = [p if p.is_dir() else _dirs(args)["runs"] / p for p in map(Path, args.runs)]
    rows, superseded, dropped = load_v3(run_dirs, _questions_for(_dirs(args)))
    if not rows:
        _err(f"error: no results in {', '.join(map(str, run_dirs))}")
        return 2
    out = Path(args.out)
    meta = {"runs": [str(p) for p in run_dirs], "superseded_units": superseded, "dropped_units": dropped}
    result = analyze_v3(rows, out, resamples=args.resamples, meta=meta)
    shown = lambda v: "n/a" if v is None else f"{float(v):.4f}"  # noqa: E731
    for t in result["tests"]:
        _out(f"{t['test']:<20} {t['model']:<18} W {t['window']} {t['family']:<9} estimate {shown(t['estimate'])} CI [{shown(t['ci_low'])}, "
             f"{shown(t['ci_high'])}] p {shown(t['p'])} holm {shown(t['p_holm'])}: {t['verdict']}")  # fmt: skip
    _out(f"wrote cells.csv ({len(result['cells'])} cells), tests.csv, tests.json and figure_data.csv to {out}")
    _out("wrote main_figure.png" if result["figure"] else "main_figure.png skipped: matplotlib is not importable (or no figure data)")
    return 0


def cmd_analyze_v4(args) -> int:
    run_dirs = [p if p.is_dir() else _dirs(args)["runs"] / p for p in map(Path, args.runs)]
    rows, superseded, dropped = load_v4(run_dirs, _questions_for(_dirs(args)))
    if not rows:
        _err(f"error: no results in {', '.join(map(str, run_dirs))}")
        return 2
    out = Path(args.out)
    result = analyze_v4(rows, out, resamples=args.resamples, meta={"runs": [str(p) for p in run_dirs], "superseded_units": superseded, "dropped_units": dropped})
    shown = lambda v: "n/a" if v is None else f"{float(v):.4f}"  # noqa: E731
    for t in result["tests"]:
        _out(f"{t['test']:<32} {t['role']:<12} {t['model']:<18} W {t['window']} {t['family']:<9} {t['condition']:<7} estimate {shown(t['estimate'])} "
             f"CI [{shown(t['ci_low'])}, {shown(t['ci_high'])}] p {shown(t['p'])} holm {shown(t['p_holm'])}: {t['verdict']}")  # fmt: skip
    _out(f"wrote cells.csv ({len(result['cells'])} cells), tests.csv, tests.json and figure_data.csv to {out} "
         f"(six confirmatory tests per model as drafted in PREREG-v4, Holm within model; the rest exploratory)")
    _out("wrote main_figure.png" if result["figure"] else "main_figure.png skipped: matplotlib is not importable (or no figure data)")
    return 0


def _questions_for(d):
    """run.json -> (the run's questions, its corpus texts), cached per corpus and version."""
    cache: dict = {}

    def get(run: dict) -> tuple[list[dict], dict[str, str]]:
        key = (run["corpus_id"], run.get("questions_version", "v1"))
        if key not in cache:
            manifest = load_manifest(run["corpus_id"], d["corpus_runs"])
            cache[key] = (load_run_questions(manifest, d["corpus_runs"], key[1]), corpus_texts(manifest))
        return cache[key]

    return get


def cmd_quick(args) -> int:
    rows = quick(_run_dir(args), _questions_for(_dirs(args)))
    if not rows:
        _err(f"error: no results in {_run_dir(args)}")
        return 2
    _out(render_quick(rows))
    return 0


def cmd_diagnose(args) -> int:
    get = _questions_for(_dirs(args))
    parts = diagnose_abstain(_run_dir(args), lambda run: get(run)[0])
    if not parts:
        _err(f"error: no run in {_run_dir(args)}")
        return 2
    _out(render_diagnose(parts))
    return 0


def cmd_spend(args) -> int:
    d = _dirs(args)
    key = KEY_LEDGERS[args.key]
    if args.action == "status" and not args.spend_file:
        return _spend_status_all(d)
    path = Path(args.spend_file) if args.spend_file else key.path(d["spend"])
    records = read_spend(path)  # raises SpendLedgerCorrupt on a broken chain
    if args.action == "verify":
        _out(f"{path}: {len(records)} records, chain verified")
        return 0
    if args.action == "resolve":
        if not args.assume_spent:
            _err("error: resolve needs --assume-spent (open reservations are closed at their worst case)")
            return 2
        lock = path.parent / "lock" if path.name == "spend.jsonl" else path.with_suffix(".lock")  # a run's fake-spend.jsonl locks fake-spend.lock
        with SpendGuard(path, run_id="resolve", pricing=PricingTable.verified_default(), key=key, resolve=True, lock_path=lock) as guard:
            closed = guard.resolve_assume_spent()
        _out(f"resolved {len(closed)} open reservations as assume_spent: {closed}")
        return 0
    status = ledger_status(key, records)  # --spend-file: that one ledger, under the --key's caps
    _out(f"{args.key} ledger {path}: {status['records']} records")
    _out(render_status(status))
    if status["open"]:
        _out(f"open reservations {status['open']}: run `spend resolve --assume-spent --spend-file {path}` before the next live run")
    return 0


def _spend_status_all(d) -> int:
    """`spend status`: every key's own ledger (caps, spent, outstanding, remaining usable, per run
    and per model), then the combined total. Reads the ledgers only."""
    statuses = []
    for profile in PROFILES:
        key = KEY_LEDGERS[profile]
        path = key.path(d["spend"])
        statuses.append(ledger_status(key, read_spend(path)))  # raises SpendLedgerCorrupt on a broken chain
        default = " (the default)" if profile == DEFAULT_PROFILE else ""
        _out(f"{profile} key{default}: ledger {path.relative_to(d['base'])}, {statuses[-1]['records']} records")
        _out(render_status(statuses[-1]))
    _out(render_combined(statuses))
    for s in statuses:
        if s["open"]:
            _out(f"{s['key_profile']} open reservations {s['open']}: run `spend resolve --assume-spent --key {s['key_profile']}` before its next live run")
    return 0


def cmd_leakscan(args) -> int:
    d = _dirs(args)
    runtime = None if args.workdir is None else [d["runs"], d["spend"], d["cache"]]
    report = scan(root=ROOT, runtime=runtime)
    _out(report.render())
    return 0 if report.ok else 1


def _key_arguments(p: argparse.ArgumentParser) -> None:
    """--key and --use-reserve: the key profile picks the API key, its spend ledger and its caps."""
    p.add_argument("--key", choices=PROFILES, default=DEFAULT_PROFILE,
                   help="the API key, with its own spend ledger and caps: research (default) or aperture (only when chosen)")  # fmt: skip
    research = KEY_LEDGERS["research"]
    p.add_argument("--use-reserve", action="store_true",
                   help=f"release the key's reserve (research: ${research.reserve_usd} of ${research.hard_max_usd}); recorded in its spend ledger")  # fmt: skip


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m context_ledger.eval", description="The context-ledger pilot harness.")
    parser.add_argument("--workdir", help="where runs/, .cache/corpus/ and .clp-spend/ live (default: the repo)")
    sub = parser.add_subparsers(dest="command", required=True)

    corpus = sub.add_parser("corpus", help="build the nested corpora, snapshot an extension's sources, or extend one (BUILD-V4)")
    corpus.add_argument("action", choices=["build", "extend", "snapshot"])
    corpus.add_argument("--seed", type=int, help="build: required; extend and snapshot: default 1")
    corpus.add_argument("--base", help="extend and snapshot: the base corpus id (snapshot: or its runs/corpus directory)")
    corpus.add_argument("--sources", help=f"extend and snapshot: the frozen source list (default {SOURCES_FILE.name})")
    corpus.add_argument("--store", help="extend and snapshot: the snapshot store (default data/corpus-src)")
    corpus.add_argument("--snapshot", help="extend: the snapshot manifest (default: the store's one manifest of --base)")
    corpus.add_argument("--estimator", choices=["auto", "bytes4-v0", "calib-v1"], default="auto")
    corpus.add_argument("--model", default="claude-haiku-5-5")
    corpus.add_argument("--rho", default=None, help="build: 10,100,1000 by default; extend and snapshot: the level, 10000 by default")
    corpus.add_argument("--window", type=int, default=corpus_mod.WINDOW)
    corpus.add_argument("--version", choices=sorted(corpus_mod.BUILDERS), default="v1", help="v1: the M4 builder; v2: v2 question density, hop files in C10")
    corpus.add_argument("--tarball", help=argparse.SUPPRESS)  # tests: a fixture tarball
    corpus.add_argument("--sha256", default=corpus_mod.TARBALL_SHA256, help=argparse.SUPPRESS)

    questions = sub.add_parser("questions", help="generate questions and episodes")
    questions.add_argument("action", choices=["build", "overlap"], help="overlap: a questions file (--version, default v4) against questions v3")
    questions.add_argument("--corpus")
    questions.add_argument("--seed", type=int, default=None, help="default 1 (3 for --version v4)")
    questions.add_argument("--episodes", type=int, default=12)
    questions.add_argument("--version", choices=["v1", "v2", "v2s", "v3", "v4"], default="v1",
                           help="v1 keeps the M4 questions (and ids); v2 writes questions-v2.jsonl; v2s is v2 with session facts, questions-v2s.jsonl; "
                                "v3 is the v2s generator with its own label, questions-v3.jsonl")  # fmt: skip
    questions.add_argument("--questions-per-episode", type=int)
    questions.add_argument("--mix")
    questions.add_argument("--probe-gap", type=int, default=8)
    questions.add_argument("--dry-run", action="store_true")

    cal = sub.add_parser("calibrate", help="fit calib-v1 against count_tokens")
    cal.add_argument("--live", action="store_true")
    _key_arguments(cal)
    cal.add_argument("--counter", choices=["fake", "count_tokens"], default="count_tokens")
    cal.add_argument("--model", default="claude-haiku-5-5")
    cal.add_argument("--seed", type=int, default=1)
    cal.add_argument("--out")
    cal.add_argument("--envelope", action="store_true", help="fit the envelope model (message and block overheads, tool definitions)")
    cal.add_argument("--tarball", help=argparse.SUPPRESS)
    cal.add_argument("--sha256", default=corpus_mod.TARBALL_SHA256, help=argparse.SUPPRESS)

    pl = sub.add_parser("plan", help="print the cost plan and its verdict")
    pl.add_argument("--measured", nargs="+", help="a smoke run directory (at the same window) to take measured figures from; with --v4, the pilot runs")
    pl.add_argument("--window", type=int, default=corpus_mod.WINDOW)
    pl.add_argument("--v4", action="store_true", help="the v4 pilot and grids (BUILD-V4, BUILD-V4B)")
    pl.add_argument("--measured-v3", nargs="*", help="with --v4: the v3 run directories to measure B3, B4 and L from")
    pl.add_argument("--max-calls-episode", type=int, default=V4_MAX_CALLS, help=f"with --v4: the episode call bound priced (default {V4_MAX_CALLS})")
    pl.add_argument("--model", choices=[HAIKU, SONNET], help="with --v4: a custom grid's model")
    pl.add_argument("--episodes", type=int, help="with --v4: a custom grid's episodes")
    pl.add_argument("--rhos", help="with --v4: a custom grid's rho list")
    pl.add_argument("--arms", help="with --v4: a custom grid's arms")
    pl.add_argument("--conditions", choices=["allowed,forced", "allowed", "forced"], help="with --v4: a custom grid's conditions (default both)")
    _key_arguments(pl)

    run = sub.add_parser("run", help="run arms over questions")
    run.add_argument("--live", action="store_true")
    _key_arguments(run)
    run.add_argument("--arms", default="B3,B4,L")
    run.add_argument("--rho", help="one rho or a comma list (default 10; with --grid, the model's grid)")
    run.add_argument("--episodes", default="1", help="the first N episodes, or an inclusive range a-b")
    run.add_argument("--questions-per-episode", type=int, help="default: every question the questions file has per episode")
    run.add_argument("--questions", choices=["v1", "v2", "v2s", "v3", "v4"], default="v1",
                     help="the questions file: v1 questions.jsonl, v2 questions-v2.jsonl, v2s questions-v2s.jsonl (session facts), v3 questions-v3.jsonl or v4 questions-v4.jsonl")
    run.add_argument("--model", default="claude-haiku-5-5", choices=["claude-haiku-5-5", "claude-sonnet-5-5"])
    run.add_argument("--budget", required=True, help="this run's cap in USD (at most what the global cap has left)")
    run.add_argument("--window", type=int, default=corpus_mod.WINDOW)
    run.add_argument("--smoke", action="store_true")
    run.add_argument("--grid", action="store_true")
    run.add_argument("--corpus")
    run.add_argument("--run-id")
    run.add_argument("--prompt", choices=sorted(PROMPTS), help="answer-format prompt version (default: the condition's; v1 reproduces the M3 smoke runs)")
    run.add_argument("--condition", choices=["allowed", "forced"], default="allowed",
                     help="allowed: abstaining is allowed (clp-v3-allowed); forced: every arm is told always to answer (clp-v3-forced); clp-v4-* with B5, L4 or L4-nogate")
    run.add_argument("--max-calls-episode", type=int, default=90, help="the model calls an episode may make, every arm alike (default 90, as v3; recorded only when changed)")

    for name in ("score", "report", "quick", "diagnose-abstain"):
        p = sub.add_parser(name)
        p.add_argument("run")
        if name == "report":
            p.add_argument("--resamples", type=int, default=10_000)

    an = sub.add_parser("analyze", help="the pre-registered analysis (PREREG sections 4-5) of one or more runs")
    an.add_argument("runs", nargs="+", help="run directories (or run ids under runs/); a unit in several runs is taken from the last")
    an.add_argument("--out", required=True, help="where to write cells.csv, tests.csv, tests.json, exclusions.csv, figure_data.csv")
    an.add_argument("--resamples", type=int, default=10_000)

    an3 = sub.add_parser("analyze-v3", help="the PREREG-v3 analysis of v3 runs (both conditions)")
    an3.add_argument("runs", nargs="+", help="run directories (or run ids under runs/); a unit in several runs is taken from the last")
    an3.add_argument("--out", required=True, help="where to write cells.csv, tests.csv, tests.json, figure_data.csv (and main_figure.png)")
    an3.add_argument("--resamples", type=int, default=10_000)

    an4 = sub.add_parser("analyze-v4", help="the v4 analysis of runs with B5, L4 and L4-nogate (BUILD-V4B decisions 4-5)")
    an4.add_argument("runs", nargs="+", help="run directories (or run ids under runs/); a unit in several runs is taken from the last")
    an4.add_argument("--out", required=True, help="where to write cells.csv, tests.csv, tests.json, figure_data.csv (and main_figure.png)")
    an4.add_argument("--resamples", type=int, default=10_000)

    spend = sub.add_parser("spend", help="inspect or resolve the spend ledger")
    spend.add_argument("action", choices=["status", "verify", "resolve"])
    spend.add_argument("--assume-spent", action="store_true")
    spend.add_argument("--spend-file")
    spend.add_argument("--key", choices=PROFILES, default=DEFAULT_PROFILE, help="the ledger verify and resolve use (status shows every key)")

    sub.add_parser("leakscan", help="scan artifacts and history for the key")

    ix = sub.add_parser("index", help="build the address index of a corpus (BUILD-V4)")
    ix.add_argument("action", choices=["build"])
    ix.add_argument("--corpus")
    return parser


COMMANDS = {"corpus": lambda a: {"extend": cmd_extend, "snapshot": cmd_snapshot}.get(a.action, cmd_corpus)(a), "index": cmd_index, "questions": cmd_questions, "calibrate": cmd_calibrate, "plan": cmd_plan, "run": cmd_run,
            "score": cmd_score, "report": cmd_report, "analyze": cmd_analyze, "spend": cmd_spend, "leakscan": cmd_leakscan,
            "quick": cmd_quick, "diagnose-abstain": cmd_diagnose, "analyze-v3": cmd_analyze_v3, "analyze-v4": cmd_analyze_v4}  # fmt: skip


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return COMMANDS[args.command](args)
    except (LiveRefused, BudgetRefused, OpenReservation, SpendError, KeyUnavailable, PricingError, CalibrationMissing, CalibrationError, EnvelopeError,
            CorpusError, QuestionError, PlanError, RunConfigMismatch, IndexError_) as exc:  # fmt: skip
        _err(f"refused: {safe_message(exc)}")
        return 2

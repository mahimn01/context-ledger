# Context ledger: The window is a cache

Companion code and data for [The window is a cache](https://mahimn.dev/essays/the-window-is-a-cache), by Mahimn Patel, November 2026.

This is a fresh, sanitized research-artifact export, not the history of the private working repository. The experiment tests keyword retrieval, rolling compaction, a source-address index, and an evidence ledger with a target/value gate under a 3,000-token working budget. Corpus level labels are not literal ratios to that budget. Results are benchmark-specific, not evidence that this verifier proves arbitrary answers true.

## Where the post's claims come from

- [Original preregistration](PREREG.md), [v3](PREREG-v3.md) and [v4](PREREG-v4.md).
- [Post-freeze bug amendment](PREREG-v4-AMENDMENT-1.md), [later ablation amendment](PREREG-v4-AMENDMENT-2.md), [further disclosure](PREREG-v4-AMENDMENT-3.md).
- [Architecture and v3 failure diagnostic](ARCH-v4.md).
- Main completed core: [cells](runs/v4/grid-analysis-c260-fix1/cells.csv), [tests](runs/v4/grid-analysis-c260-fix1/tests.csv), [analysis provenance](runs/v4/grid-analysis-c260-fix1/tests.json).
- Later Sonnet gate-off ablation: [cells](runs/v4/grid-analysis-c260-nogate1/cells.csv), [tests](runs/v4/grid-analysis-c260-nogate1/tests.csv).
- [Graded per-question records](runs/v4/per-question), plus raw final-answer records and run configurations under `runs/v4-grid-*`. These are not conversation/call transcripts.
- Earlier work: [first grid](runs/analysis-m4/cells.csv), [v2 pilot cells](runs/v2-pilot-analysis/pilot_cells.csv), [v2 per-question errors, including ST_ATIME](runs/v2-pilot-analysis/pilot_errors.csv).

The blog counts actual arm totals, not the discordant-pair columns in paired statistical tests. Unlabelled errors are `wrong + false_answer - unverified_guess_wrong - unverified_guess_false_answer`.

## Reproduce the archived statistics without spending money

Python 3.10+ is sufficient for the numerical analysis. From the repository root:

```sh
PYTHONPATH=src python tools/reproduce_v4.py --analysis core --out reproduced/core
PYTHONPATH=src python tools/reproduce_v4.py --analysis latest --out reproduced/latest
PYTHONPATH=src python -m unittest discover -s tests -p test_public_credentials.py
```

These commands use released, already-graded question records and 10,000 episode-bootstrap resamples with seed 0. They make no model/API calls. Matplotlib is optional for figures. The original scoring, generation, indexing and evaluation implementations are in `src/context_ledger/`; corpus construction and live-grid commands are documented in the preregistrations and `scripts/v4/`.

The source corpora and full call traces are deliberately omitted. Re-evaluating the model, rather than recomputing statistics, requires rebuilding the named public source corpus, installing the optional Anthropic SDK, supplying your own explicitly configured credential and authorizing your own budget. Do not run the live grid accidentally. Recorded provider IDs and prices are historical, not a promise of present availability.

The public export replaces the private OS credential adapter with an environment-only adapter. `CONTEXT_LEDGER_RESEARCH_API_KEY` configures the research profile; `CONTEXT_LEDGER_LEGACY_API_KEY` configures the older, separately budgeted profile still named `aperture` in historical records. No platform credential integration, credential values or original personal storage are included. The original safety/budget guard remains in place. See [SANITIZATION.md](SANITIZATION.md).

## License

The implementation retains its original MIT license. Third-party source material keeps its upstream rights; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Privacy redactions are documented, and original local history is not distributed.

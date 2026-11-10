# PREREG-v4 Amendment 2: the Sonnet L4-nogate arm, so the gate ablation exists on both models

Dated Nov 5, 2026, 1:49 AM EST. Committed by Mahimn Patel before any Sonnet L4-nogate call: no run
id ending in -nogate1 exists at this commit, and scripts/v4/run-sonnet-nogate.sh refuses to start
until this file is committed as it is (it prints this file's commit and time first). Everything not
stated here is as frozen in PREREG-v4 (e903b45) and amended by Amendment 1 (0918dc1).

Code: scripts/v4/run-sonnet-nogate.sh with its tests (96b9b70) and the analysis tests (ab6aff8).
src/ is unchanged since Amendment 1 (0918dc1).

## Why, and when it was decided
PREREG-v4 section 4 runs L4-nogate on Sonnet "only if the budget allows", and section 9 decides it
after the core Sonnet grid: it runs if the research spend then left covers its estimate x 1.2.
Amendment 1 capped total research-key spend at $150 and did not run it. The core Sonnet grid and
Amendment 1's rerun are done (its last run ended Nov 4, 03:13 EST). At this commit the research key
has spent $122.345727, $0 outstanding (`eval spend status`), so $27.654273 is left under $150. The
arm's estimate x 1.2 is $12.94 (below). That fits, so section 9's condition holds against Amendment
1's stricter $150, as it does against section 9's own $180 ($57.65 left). Mahimn approved running
the arm on Nov 5.

Disclosure: this is decided after the results in runs/v4/grid-analysis-c260-fix1 (Nov 4, 03:13 EST)
were computed and seen. They include Haiku's gate ablation (exploratory): L4 had fewer misses,
confident errors and forced unlabelled fabrications than L4-nogate. Section 9's condition is about
budget only and does not depend on any outcome. No Sonnet L4-nogate row exists yet. The confirmatory
families are not touched (Analysis, below). This amendment adds an arm and changes nothing
Amendment 1 set.

## Design (this arm only; scripts/v4/run-sonnet-nogate.sh)
- Model claude-sonnet-5-5, arm L4-nogate only. Its mechanism is l-v4-nogate: L4 with the answer gate
  off, and so no gate-triggered repair. Prompts (clp-v4), tools, pre-resolution and forced handling
  are L4's (PREREG-v4 section 2 decision 3, section 5).
- All 16 cells of the Sonnet grid: allowed and forced x rho 10, 100, 1000, 10000 x episodes 1-4 and
  5-8 (questions v4 episodes 1-8, the episodes Sonnet's other arms ran). One run per cell, with
  L4-nogate alone in it.
- W = 3,000; 260 model calls per episode; --key research; never --use-reserve.
- Questions v4: questions-v4.jsonl, sha256 ad98051173a38fa13ba053f00f02d251e6de7d80c8eb7adcf59b0ec050d4b4c9.
- Corpus cf9bb5aff4ed6:
  - runs/corpus/cf9bb5aff4ed6/manifest.json, sha256
    cdbc6aa6cf2a997336641c4095945dc694192549126a071de5f9f61d2ae91c12;
  - the address index every v4 grid run used, .cache/corpus/cf9bb5aff4ed6/address-v4.sqlite, sha256
    e27499598413619c531bf428a12a41930bdb16e58b30a1a874b8c40738c8206a;
  - the source snapshot as in PREREG-v4 (MANIFEST-cf9bb5aff4ed6.tsv, sha256
    f763f91c0ebb4cd05a6612c792223908a44d39324433d7cfd0337e79c47e3108).

  The script checks the manifest, the index and the questions file against these sha256s before
  anything is billed.
- Harness: the code at this commit. It records harness_patch
  "v3.1-empty-text-cache-control+v4.1-thinking-binding+cut-tool-use" (Amendment 1's fixes) in
  run.json and in every row. That is the harness of Sonnet's L4 units (the -fix1 runs). The script
  refuses to start if src/ or the script itself differs from the commit.
- Run ids v4-grid-sonnet-<cond>-r<rho>-e<a>-<b>-c260-nogate1, in grid.sh's balanced order:
  episodes 1-4 over rho 10, 100, 1000, 10000 (allowed, then forced, at each rho), then episodes 5-8
  the same way. Sequential, one live run at a time.
- Logs: each run to runs/v4/<id>.log (exit status from zsh ${pipestatus[1]}, INCOMPLETE flagged),
  the driver to runs/v4/nogate-driver.log, the estimate to runs/v4/nogate-plan.log.
- A complete -nogate1 run is skipped on a re-run. An incomplete one resumes under its own id, only
  after its cause is looked at.
- Nothing else runs: no other arm, model or episode, and no spare episode.

Differences from Haiku's L4-nogate (threats; they bear on the exploratory reading only):
- Haiku ran L4-nogate in its grid runs, after L4 in the same run. Sonnet's runs in separate runs,
  one to two days after Sonnet's L4 (the -fix1 runs, Nov 3-4). The model, prompts, questions and
  harness are the same; the time of the calls is not.
- Haiku's L4-nogate units ran on the harness before Amendment 1's patch, except forced rho 10000
  (from -fix1 runs). With thinking off (Haiku), the requests are byte-identical with and without
  the patch (Amendment 1).

## Cost, caps and the stop rule
Estimate: `eval plan --v4 --measured`, from completed units only, over the v4 Sonnet and Haiku grid
runs, their -fix1 runs and both pilots (runs/v4/nogate-plan.log when the script runs). No Sonnet
L4-nogate unit exists yet, so the plan prices it as Haiku's measured L4-nogate times the v3
Sonnet/Haiku cost ratio of L (19.0 allowed, 17.9 forced).

| per episode-arm unit (26 questions) | allowed | forced |
|---|---|---|
| Sonnet L4-nogate (Haiku's measured x the v3 ratio) | $0.139723 | $0.197330 |
| for reference: Sonnet L4, measured | $0.161470 | $0.285530 |
| for reference: Haiku L4-nogate and L4, measured | $0.007357 and $0.009059 | $0.011048 and $0.018716 |

| item | expected | x 1.2 |
|---|---|---|
| Sonnet L4-nogate: 8 episodes x 4 rho x 2 conditions | $10.785674 | $12.942809 |

A cross-check (not used for anything): Sonnet's measured L4 times Haiku's measured L4-nogate/L4
ratio gives $9.59 ($11.51 x 1.2). The plan's figure is the more conservative of the two.

Per-run --budget: $1.15 allowed and $1.60 forced. That is 2x the expected 4-episode run, rounded up
to $0.05, the plan's own suggestion. The 16 runs hold $22.00 of caps in all. Expected research
spend after the arm: about $133.13.

STOP RULE: total research-key spend stays at or under $150 (Amendment 1's cap, kept).
- Before each run, halt if the research key's spend plus that run's --budget would pass $150. The
  spend is spent + outstanding, from its own ledger .clp-spend/research/spend.jsonl, read to the
  micro-dollar.
- A halt ends the arm where it is, and no later run is tried. A partial arm is therefore a prefix
  of the balanced order: once the first eight runs are done, every rho and condition has episodes
  1-4. A partial arm is reported as partial, with its dropped units.
- A run that stops on API errors, or exits non-zero (a crash or a failed leak scan), halts
  everything. Nothing more is billed until it is looked at; the read-only analysis still runs over
  what completed.
- A run that stops at its own cap is flagged INCOMPLETE, and the arm goes on. Its completed
  episodes are analysed and the episode it stopped in is dropped.
- At these figures the rule cannot bind on the arm's own spend: $122.35 + $22.00 = $144.35, even if
  every run spent its whole cap. It binds only if other research-key spend enters the ledger
  meanwhile. Had the estimate x 1.2 been more than what is left under $150, the arm would still
  start (the script prints a NOTE), and the rule would halt it.

## Analysis
`eval analyze-v4`, unchanged, into runs/v4/grid-analysis-c260-nogate1. The runs are given in this
order: the original grid runs, then the -fix1 runs, then the -nogate1 runs. runs/v4/grid-analysis-c260
and -c260-fix1 are kept as they are. Amendment 1's unit rules apply. A unit is (model, W,
condition, arm, rho, episode).
- A unit a run stopped in, or short of rows, is dropped and listed in `dropped_units`.
- A unit complete in several runs is taken from the run given last, and the earlier is listed in
  `superseded_units`.
- Infrastructure error rows are left out.

The -nogate1 runs hold only Sonnet L4-nogate units, which no earlier run has, so they supersede
nothing.

The Sonnet gate-ablation comparisons are SECONDARY/EXPLORATORY. They are not part of the six
confirmatory tests. For both models (Haiku primary, Sonnet secondary) the six stay as already
reported in runs/v4/grid-analysis-c260-fix1, with Holm within each model's six. analyze-v4 computes
the six from L4, B3 and B4 rows alone. So adding L4-nogate rows leaves every confirmatory row, and
every other existing row, identical; tests/test_analyze_v4.py checks this. The code at this commit
also reproduces runs/v4/grid-analysis-c260-fix1 byte for byte from the same runs.

They use the same metrics and tests as Haiku's gate ablation: raw p, no Holm, rho pooled,
bootstraps over episodes (10,000 resamples, seed 0). The McNemar tests are exact, item-paired and
one-sided for L4 fewer.
- ACC-L4vsL4-nogate (allowed): answerable items not answered correctly.
- CE-L4vsL4-nogate (allowed): confident errors over all items. A confident error is a wrong answer
  to an answerable item, or any answer to an unanswerable one.
- UFAB-L4vsL4-nogate-forced (forced, every rho): unlabelled fabrications, the confident errors not
  flagged unverified_guess.
- Per arm: TOST-ACC-L4-nogate-allowed and -forced (the OLS slope of accuracy on log10 rho, TOST
  within +/-2 pp per decade), and SLOPE-FAB-L4-nogate-allowed and -forced.
- The cells per rho and subgroup: accuracy, coverage, confident errors, fabrication, look-alike and
  stale errors, unverified guesses, repairs and cost. Primary variant only, as for Haiku's
  L4-nogate: the gate is off, so there is no gated variant.

Sonnet's gate ablation is read beside Haiku's as a replication; both are labelled exploratory. If
the arm is partial, each model's ablation is reported with the episodes it has. Nothing else
changes.

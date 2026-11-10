# PREREG-v4 Amendment 1: two harness bugs that stopped L and L4, their fixes, and the rerun of the lost units

Dated Nov 3, 2026, 4:56 PM EST. Committed by Mahimn Patel before any rerun call (no run id ending
in -fix1 exists at this commit). Everything not stated here is as frozen in PREREG-v4 (e903b45).

## What happened
Sonnet (claude-sonnet-5-5), all 16 grid runs (runs/v4-grid-sonnet-*-c260). 15 runs stopped in L
with `stopped: {arm: L, reason: api_errors}` on repeated error rows, so L4, which runs after L,
never started in them. The 16th, forced r10 e5-8, completed, with two
error rows in L4 episode 7. In all: 46 L and 2 L4 error rows. L produced 568 of its 1,664 rows
and L4 104. Every error was the API's HTTP 400 invalid_request_error "messages.N.content.M: Invalid
`signature` in `thinking` block. The block is bound to a different conversation. Remove the block,
or set ... Content before this block was rewritten since it was created, first at
`messages.X.content.Y`." B3, B4 and B5 are complete (1,664 rows each, no error). The Sonnet cost
pilot's forced run (runs/v4-pilot-sonnet-forced-c260, before the freeze) already had two L4 error
rows of this kind; they were not noticed before the freeze.

Haiku (claude-haiku-5-5), two runs, not one: v4-grid-haiku-forced-r10000-e1-5-c260 stopped in L
episode 2 (10 of 26 rows), and v4-grid-haiku-forced-r10000-e6-10-c260 stopped in L episode 9 (12 of
26 rows). Each had three error rows, all InvalidWindow ("message 3: a tool_use is not answered by
the next message"). That is the harness's own check before sending, so no request was sent and
nothing was billed for those rows. So L episodes 3-5 and 10, and L4 and L4-nogate in episodes 1-10,
never ran at Haiku forced rho 10000. Every other Haiku run is complete.

## Causes and fixes
1. Sonnet, commit 0f4d683. Sonnet 5.5's lowest thinking setting (between_tools) returns a
   thinking block with tool calls, and L passes it back in its assistant turn. Read-only replays of
   the 18 ledgers with a refused call (17 in the grid, 1 in the Sonnet pilot) show one cause every
   time: L's backstop folded earlier turns into a fold
   line between the call that returned the block and the next call, so the block went back after
   rewritten content. The same l-v3 code sent 37 such blocks in the v3 Sonnet grid (Nov 1) and the
   API took every one. From Nov 2 it refused all 50 and took the 22 that followed unchanged content.
   The API's rule changed between the two grids; the harness did not. The fix: the client records
   each thinking block's binding (the digest of the messages before it) in its model_call, and
   the request builder sends the block back only while that digest still holds. Otherwise the
   block is left out, never altered. P15 still rebuilds every request from the ledger. The fake
   server now binds thinking blocks the same way and answers a rebound one with the API's 400
   word for word (tests/test_thinking_binding.py, from a real error row). L4 and L4-nogate use
   the same request path and get the same fix. B3, B4 and B5 do not use it.
2. Haiku, commit 3431b62. A response cut at max_tokens held 12 ctx_refetch calls. The turn ended
   with them unanswered, so every later request of the episode was invalid. These are the only two
   such responses in every v3 and v4 ledger. L's and L4's turns now append an error tool_result
   ("Not run: your response stopped (max_tokens) before this tool call could run.") for each tool
   use of a response that ends the turn at max_tokens or a refusal.

Both fixes are deviations from the frozen mechanisms, disclosed here. Fix 1 changes what Sonnet
sees only where the API now refuses the request: no accepted v4 request had a rebound block, but
the v3 Sonnet grid's L sent 37 that the API then took, so a v3 Sonnet L unit run today would differ
there (v3's runs and analysis are not touched). Fix 2 changes what the model sees only where the
episode could not go on; no completed unit of any v3 or v4 run reached it. With thinking disabled
(Haiku), the requests are byte-identical with and without both fixes
(tests/test_thinking_binding.py), and v3 stays byte-identical (tests/test_v3_reproducible.py). The
mechanism labels stay l-v3, l-v4 and l-v4-nogate. New runs with a v4 arm record harness_patch
"v3.1-empty-text-cache-control+v4.1-thinking-binding+cut-tool-use" in run.json and in every L, L4
and L4-nogate row. No prompt, tool, gate, question, corpus, window, model, bound or grader
changes.

Latent, unchanged:
- B5's request fitting (drop the lowest-ranked chunk, then elide the oldest tool result) could
  rewrite content before a Sonnet thinking block within a question. No B5 row has an error
  (1,664 Sonnet rows under the enforcing API).
- B4 would leave a cut response's tool uses unanswered. No B4 transcript has such a response.

The fix assumes the API accepts a request with the rebound block removed, as its own 400 says
("Remove the block"). This cannot be checked offline. The rerun script therefore halts everything
as soon as any run stops on API errors, so at most one run's cap is spent if the assumption fails.

## Rerun (only the lost units; scripts/v4/rerun-sonnet.sh)
Same questions v4 (sha256 ad98051173a38fa13ba053f00f02d251e6de7d80c8eb7adcf59b0ec050d4b4c9), corpus
cf9bb5aff4ed6, W = 3,000, models, condition prompts, 260 calls per episode, --key research, never
--use-reserve.
- Sonnet: arms L and L4 only, in all 16 cells (allowed and forced x rho 10, 100, 1000, 10000 x
  episodes 1-4 and 5-8). Run ids v4-grid-sonnet-<cond>-r<rho>-e<a>-<b>-c260-fix1, in grid.sh's
  balanced order (first half over every rho and condition, then the second). This reruns the 12
  L and 3 L4 Sonnet units that had completed too. They ran on the unpatched harness, and rerunning
  them means all Sonnet L and L4 data come from one harness. This was decided before any rerun
  outcome.
- Then the lost Haiku forced rho-10000 units, in the same order: L episodes 2-5 and 9-10, and L4
  and L4-nogate episodes 1-5 and 6-10 (run ids v4-grid-haiku-forced-r10000-e{2-5,1-5,9-10,6-10}-c260-fix1).
  The build brief named only the e6-10 run. e1-5 stopped from the same cause, and without it no
  Haiku L4 forced unit at rho 10000 would exist, so it is included. Complete Haiku units are not
  rerun.
- Baselines B3, B4 and B5 are not rerun; their grid units are reused unchanged. Sonnet L4-nogate
  is not run by this amendment.
- Each run logs to runs/v4/<id>.log (exit status from zsh ${pipestatus[1]}, INCOMPLETE flagged),
  the driver to runs/v4/rerun-driver.log. A run that stops is reported, and is resumed under its
  own id only after its cause is looked at. A unit that fails again is reported as missing.

## Budget
Research key at this commit: $78.566027 spent, $0 outstanding, of the $180 usable (`eval spend
status`). This amendment caps the rerun so total research-key spend stays at or under $150.
STOP RULE: before each run, halt if the research key's spend (spent + outstanding) plus that run's
--budget would pass $150. While Sonnet runs remain, the four Haiku caps ($1.85) are held back too,
so the primary family is completed even if Sonnet halts. A Sonnet halt ends the Sonnet phase in
balanced order, and the Haiku units still run.

Estimate (`eval plan --v4 --measured`, completed units only: not the unit a run stopped in, nor
one with a partial or API-error row, commit 6062cc9; runs/v4/rerun-plan.log when the script runs):

| item | measured from | expected | x 1.2 |
|---|---|---|---|
| Sonnet L, L4: 8 episodes x 4 rho x 2 conditions | completed Sonnet grid and pilot units | $42.43 | $50.91 |
| Haiku forced rho 10000: L4, L4-nogate x 10 episodes | Haiku grid and pilot | $0.69 | $0.83 |
| Haiku forced rho 10000: L x 6 episodes | Haiku grid and pilot | $0.21 | $0.25 |
| total | | $43.33 | $51.99 |

Per episode-arm unit (Sonnet, allowed / forced): L $0.382 / $0.534, L4 $0.160 / $0.250. These
rest on 4 / 8 completed L units and 1 / 3 completed L4 units, so the Sonnet L4 allowed figure is
thin. Per-run --budget (2x expected): Sonnet $4.35 allowed, $6.30 forced; Haiku $0.30 (L e2-5),
$0.70 (L4 and L4-nogate e1-5), $0.15 (L e9-10), $0.70 (L4 and L4-nogate e6-10). Expected research
spend after the rerun: about $121.90. The caps sum to $87.05, more than the $71.43 left under
$150, so the stop rule binds only if costs run well above the estimate.

## Analysis
`eval analyze-v4`, unchanged, over the original grid runs given first and the -fix1 runs after,
into runs/v4/grid-analysis-c260-fix1. A unit is (model, W, condition, arm, rho, episode).
- A unit a run stopped in, or short of rows, is dropped and listed in `dropped_units`.
- A unit complete in several runs is taken from the run given last, and the earlier is listed in
  `superseded_units`.
- Infrastructure error rows are left out, as before.

The confirmatory family is unchanged:
- the six tests per model and W;
- margins: non-inferiority 3 pp, TOST +/-2 pp per decade;
- alpha .05 one-sided, Holm within each model's six;
- 10,000 resamples, seed 0;
- Haiku primary, Sonnet secondary.

The Haiku confirmatory results were already computed before the rerun (runs/v4/grid-analysis-c260,
Nov 3, 09:03 EST). All six were supported, non-inferior or equivalent:
- NI-ACC-L4vsB3 +0.289;
- CE-L4vsB3 -0.023;
- CE-L4vsB4 -0.037;
- TOST-ACC-L4-allowed slope +0.0027 per decade;
- TOST-ACC-L4-forced -0.0023;
- UFAB-L4vsB4-forced -0.146;
- Holm p <= 0.0071 for each.

Two of them were computed on an incomplete design, and only because of the two stopped Haiku runs.
Neither run left any Haiku L4 forced unit at rho 10000. So TOST-ACC-L4-forced covered rho 10, 100
and 1000 only, and UFAB-L4vsB4-forced (rho >= 100) pooled rho 100 and 1000 only (520 items, not
780). The allowed tests had every rho. Exploratory rows missing rho 10000 for the same reason:
SLOPE-FAB-L4-forced and the L4-nogate forced TOST and slope. L forced at rho 10000 had 4 of its 10
episodes.

The rerun's purpose is to complete the pre-registered design, not to revisit a result. The Haiku
family is recomputed once, on the completed units, and both versions are reported side by side:
the earlier one labelled as computed on the incomplete design, any verdict that differs named as
such. The Sonnet family, not run so far (no paired L4 rows), is computed on the completed units
for the first time. Nothing else changes.

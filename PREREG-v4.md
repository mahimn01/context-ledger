# PREREG-v4: an addressable, verified working set (L4) under context pressure up to rho 10,000

Status: FROZEN by the commit of this file by Mahimn Patel on Nov 2, 2026, after the pilot filled
section 9; the first grid call
(calls.jsonl of the first grid run) must be later than that commit. Code under test: branch `v4`
(mechanisms l-v4, l-v4-nogate, b5-v4; v3's B3, B4 and l-v3 unchanged and byte-for-byte
reproducible, tests/test_v3_reproducible.py). Analysis code: `eval analyze-v4`, run unchanged;
anything not in section 6 is exploratory and labelled so.

## 1. What came before (reported findings, kept as such)
- v3 grid (Haiku 5.5, W = 3,000, questions v3 episodes 1-10, rho 10/100/1000): L (l-v3) had the
  fewest confident errors but the lowest accuracy; its non-answers were mostly running out of
  rounds and of the 90-call episode bound, and the question being folded out of the window in
  turns with a note, not gate refusals (ARCH-v4.md section 1).
- v4 was built offline against those findings (ARCH-v4.md): an address index outside the window,
  pre-resolution of the asked target, gate refusals turned into targeted repair, the question
  protected, a nested extension level to rho 10,000. No v4 arm has run live.

## 2. Decisions fixed before the pilot (BUILD-V4B)
1. Episode call bound: 260 model calls per episode (26 questions x 10) for EVERY arm, in the
   pilot and the grid; recorded in each run.json as max_calls_episode.
2. Questions: fresh items only (section 4).
3. Arm L4-nogate (mechanism l-v4-nogate): L4 with the answer gate off and so no gate-triggered
   repair; Tier 0/1, pre-resolution, tools, prompts (L4's) and forced handling otherwise L4's
   (forced: the last candidate is emitted as unverified_guess; rows say gate "off").
4. The L vs L4 contrast (exploratory) is primarily on corpus questions (the asked target is not a
   session note); all turns are secondary.
5. The confirmatory family (section 6).
6. The extension corpus is built only from a committed, content-addressed source snapshot.
7. Pilot and grid scripts: scripts/v4/pilot.sh and scripts/v4/grid.sh (sections 7-8).

## 3. Questions
Q1. Does L4 keep B3's accuracy (non-inferior) while making fewer confident errors than B3 and B4?
Q2. Is L4's accuracy flat across three orders of magnitude of rho, allowed and forced?
Q3. Forced to answer, does L4 make fewer UNLABELLED fabrications than B4 at high rho?
Q4 (exploratory). What do the address index (B5 vs B3), the gate (L4 vs L4-nogate) and the v4
harness (L4 vs L) each contribute, and at what cost?

## 4. Design
- Corpus cf9bb5aff4ed6: the nested extension of ca8e01d67009c (corpus seed 1, built at W =
  10,000; C10/C100/C1000 are the base's files byte for byte) with level C10000 of 8,425
  permissively licensed site-packages files as distractors (extension seed 1, frozen source list
  src/context_ledger/eval/ext_sources_v4.json). Built by `corpus extend` from the snapshot
  data/corpus-src/MANIFEST-cf9bb5aff4ed6.tsv (sha256
  f763f91c0ebb4cd05a6612c792223908a44d39324433d7cfd0337e79c47e3108; objects gitignored); a
  rebuild reproduced the id and every file, the chunk index and the perturbation records byte for
  byte. The address index (`index build`) records each level's Merkle root in run.json.
- rho in {10, 100, 1000, 10000}. Effective rho at W = 3,000, with calib-v1's conservative token
  counts: about 34, 362, 3,343 and 33,335. CAVEAT: those counts are calib-v1's upper envelope; its
  point estimates are about 58% of them (effective rho about 20, 211, 1,944 and 19,403). The four
  levels span three orders of magnitude either way.
- W = 3,000 for every run. Conditions allowed (clp-v4-allowed) and forced (clp-v4-forced); B3, B4
  and L get v3's exact prompts.
- Arms: B3 (BM25 RAG), B4 (compaction), B5 (B3 + pre-resolution and address tools, no gate), L
  (l-v3), L4 (l-v4) everywhere; L4-nogate on Haiku (on Sonnet only if the budget allows, section 9).
- Questions v4: `questions build --version v4 --seed 3 --episodes 13` on cf9bb5aff4ed6, file
  questions-v4.jsonl, sha256 ad98051173a38fa13ba053f00f02d251e6de7d80c8eb7adcf59b0ec050d4b4c9
  (grid.sh refuses any other). 26 items per episode, v3's mix: 18 answerable corpus questions (6
  fold-probe pairs), 2 unanswerable look-alikes (near-miss names absent from all 9,281 files), 4
  answerable session probes (2 after an update: stale probes) and 2 unanswerable session probes;
  session values occur nowhere in the extended corpus; confusers counted at all four levels.
- Overlap filter (no item reused): the reference is questions v3 (ca8e01d67009c, sha256
  d6da3013c0640d3d03b0a3d3201a223f0de0d0e69d018a038b958d5235dd5b58) episodes 1-10 (v3 grid),
  11-12 (v4 pilot) and 19 (v3 smoke). An item overlaps if its target identifiers (every backticked
  name; for a session probe its service and attribute) and gold answer equal a reference item's,
  or its question text does after lower-casing and collapsing whitespace. Overlapping candidates
  never enter the pools (237 kept out: constant 87, default 71, raises 37, locate 33, default_ref
  6), a near-miss that overlaps is redrawn, session probes use only pairs no reference probe
  asked. The filter alone left 6 episodes (< 10), so the pools took extras from every .py file of
  C10 for names NO questions v3 item asks about, in v3's exact templates: unperturbed upper-case
  module constants (gold the source literal, no upstream answer) and unrenamed top-level locate;
  extras come after the v3 generator's candidates in each file. Result: 13 fresh episodes, 0 items
  overlapping the reference (`questions overlap`). Haiku uses episodes 1-10, Sonnet 1-8, 11-13 are
  spares (used only to replace an episode lost to an infrastructure failure, never after looking
  at outcomes).
- Episodes 1-10: 260 items; 220 answerable (180 corpus: constant 118, locate 35, raises 14,
  default 13; 40 session), 40 unanswerable (20 near-miss, 20 session); 20 stale probes; 78
  answerable items with confusers at some rho. Corpus items by origin: v3 generator 111,
  unperturbed constants 66, unrenamed locate 3. Episodes 1-8: 208 items; 176 answerable (144
  corpus, 32 session), 32 unanswerable; origins 95 / 47 / 2.
- Models: claude-haiku-5-5 (primary family, episodes 1-10) and claude-sonnet-5-5 (secondary
  family, episodes 1-n, n in {6, 8} by the sizing rule of section 9; never fewer arms).

## 5. Forced condition and labels (pre-specified)
- B3, B4, B5: answer calls as in v3; an UNANSWERABLE stays an abstention (forced_violation).
- L: v3's forced handling (forced_blocked fallback; forced_noanswer).
- L4: an UNANSWERABLE gets v3's one best-guess message; before the last round a blocked candidate
  gets one more repair ("final") if one is left; if the turn still ends blocked, the row's answer
  is the last candidate, flagged unverified_guess (the ledger records an abstention). L4-nogate:
  the same, with only Tier 0/1 able to block.
- The grader (eval/score.py, unchanged) grades every answer alike, unverified guesses included.

## 6. Confirmatory tests (as implemented in `eval analyze-v4`; alpha .05, one-sided)
Per model (and W), six tests, Holm-corrected within the model: Haiku is the primary family,
Sonnet its own family, labelled secondary. Items are (rho, question) pairs both arms ran, rho
pooled unless stated; bootstraps resample episodes (10,000 resamples, seed 0).
1. NI-ACC-L4vsB3 (allowed): L4's accuracy is non-inferior to B3's with margin 3 pp. Estimate: the
   item-paired accuracy difference acc(L4) - acc(B3) on the answerable items both ran; its
   episode-cluster bootstrap CI; p = the share of resamples with difference <= -0.03.
   Non-inferior if the Holm-adjusted p < .05.
2. CE-L4vsB3 (allowed): L4 makes fewer confident errors than B3. A confident error is a wrong
   answer to an answerable item or any answer to an unanswerable one (v3's fabrication, over all
   items). Exact item-paired McNemar, one-sided.
3. CE-L4vsB4 (allowed): the same against B4.
4. TOST-ACC-L4-allowed: L4's accuracy is flat across rho. The OLS slope of accuracy (per-episode
   accuracy, mean over episodes per rho) on log10 rho; TOST equivalence with margin +/-2 pp per
   decade: p = max(share of resamples <= -0.02, share >= +0.02); equivalent if Holm p < .05.
5. TOST-ACC-L4-forced: the same under the forced condition (unverified guesses graded as answers).
6. UFAB-L4vsB4-forced (forced, rho >= 100): L4 makes fewer UNLABELLED fabrications than B4. An
   unlabelled fabrication is a confident error not flagged unverified_guess (B4 flags none); exact
   item-paired McNemar, one-sided. Reported alongside (exploratory): the same count with L4's
   unverified guesses included.
Verdicts: supported / not detected (non-inferior / not shown; equivalent / not shown); an effect
in the opposite direction is reported as such.

## 7. Exploratory analyses (labelled so; raw p)
The L4/B4 median per-question cost ratio with its 95% CI, descriptive against 1.5x (both
conditions); B5 vs B3 (the index effect: misses and confident errors); L4 vs L4-nogate (the gate
effect: misses, confident errors, forced unlabelled fabrications); L vs L4 on corpus questions
(primary contrast) and on every turn (secondary), misses and confident errors; TOST and
fabrication slopes of every arm; CE-L4vsB3/B4 on look-alike and stale items; every per-rho and
subgroup cell (corpus, session, origin: v3 generator vs unperturbed extras).

## 8. Runs, order, caps and the stop rule
- Pilot (scripts/v4/pilot.sh; questions v3 episodes 11-12, not in any test): Haiku, all six arms,
  rho 10, 1000, 10000, both conditions; two runs (one per condition), --budget $2.50 each ($5.00
  in all; the pilot's limit is $15).
- Sonnet cost pilot (scripts/v4/sonnet-pilot.sh, added before freezing, cost only, not in any test):
  Sonnet, arms B5 and L4, questions v3 episode 11, rho 10000, both conditions, --budget $3.00 each;
  run ids v4-pilot-sonnet-<cond>-c260. Run after the Haiku pilot exited, never concurrently.
- Grid (scripts/v4/grid.sh, run by scripts/v4/grid-chain.sh: Haiku, then Sonnet core, sequentially): one run per (condition, rho, half of the episodes), all the model's
  arms in it; run ids v4-grid-<model>-<cond>-r<rho>-e<a>-<b>-c260. Order: Haiku episodes 1-5
  over all (rho, condition) runs, then Haiku 6-10, then Sonnet 1-4, then Sonnet 5-8, so a halted
  grid is balanced. Runs are sequential (one live run at a time).
- Per-run --budget (`eval plan --v4`: 2x a run's expected cost): Haiku $1.25 allowed, $1.55
  forced (16 runs, $22.40 of caps); Sonnet with five arms $16.30 allowed, $19.05 forced (16 runs).
  Re-derived from the pilot (section 9): kept as above (the measured 2x caps are lower: Haiku
  $1.00/$1.30, Sonnet $13.95/$17.85; the larger caps above are kept so a run is less likely to stop
  INCOMPLETE; the stop rule still bounds total spend).
- Every live command: --key research, --max-calls-episode 260, never --use-reserve.
- STOP RULE: before each run, halt if the RESEARCH key's spend (its own ledger,
  .clp-spend/research/spend.jsonl: spent + outstanding) plus that run's --budget would exceed
  $180, the research key's usable cap ($200 less its $20 reserve). The Aperture key is frozen at
  $26.27 and is not counted (equivalently, a combined cap of $206.27). Every grid run is billed to
  the research key. (Changed before any grid call from the BUILD-V4B draft, which counted the
  combined spend against $180 and so would have left only about $153 usable.) A run that hits its own cap stops INCOMPLETE; its completed episodes are
  analysed and the incomplete one dropped (analyze-v4's unit rules); a failed or incomplete run is
  resumed with the same run id. No outcome-based stopping; no episode added after looking at
  outcomes.

## 9. Cost
Estimates before the pilot (`eval plan --v4`, from the v3 grid's measured costs at W = 3,000,
scaled to 260 calls per episode; L4 and L4-nogate at L's cost, B5 at twice B3's; Sonnet's costs
measured at about 20x Haiku's per arm):

| item | expected | x 1.2 |
|---|---|---|
| pilot (6 arms, Haiku, 2 episodes, 3 rho, 2 conditions) | $1.66 | $2.00 |
| grid Haiku (6 arms, 10 episodes, 4 rho, 2 conditions) | $11.09 | $13.30 |
| grid Sonnet (5 arms, 8 episodes, 4 rho, 2 conditions) | $141.17 | $169.40 |
| grid Sonnet, L4-nogate alone | $33.70 | $40.44 |

Measured in the pilot (runs/v4/plan-v4-measured.log and `eval spend status`, Nov 2, 06:36 EST):

| measured cost per episode-arm (26 questions) | Haiku allowed | Haiku forced | Sonnet allowed | Sonnet forced |
|---|---|---|---|---|
| B3 | $0.0074 | $0.0081 | ($0.149) | ($0.167) |
| B4 | $0.0367 | $0.0374 | ($0.776) | ($0.795) |
| B5 | $0.0080 | $0.0081 | ($0.161); Sonnet pilot $0.144 | ($0.167); Sonnet pilot $0.148 |
| L | $0.0231 | $0.0379 | ($0.438) | ($0.708) |
| L4 | $0.0113 | $0.0211 | ($0.215); Sonnet pilot $0.160 | ($0.394); Sonnet pilot $0.282 |
| L4-nogate | $0.0097 | $0.0129 | ($0.185) | ($0.241) |

Parenthesised Sonnet figures are the Haiku pilot x the v3 Sonnet/Haiku cost ratio (18.7-21.2), as
`plan --v4 --measured` prices them; the Sonnet cost pilot (one episode, rho 10000) measured B5 and
L4 at 75-89% of those figures, so the plan's Sonnet estimates are used unchanged (conservative).

- Pilot spend: Haiku pilot $1.330250 (allowed $0.576931, forced $0.753319); Sonnet cost pilot
  $0.735141 (allowed $0.304675, forced $0.430466); in all $2.065391 on the research key.
  Research-key spend at freeze: $2.319980 (incl. $0.254590 of v3 L reruns before v4), outstanding
  $0. Aperture key frozen at $26.267784. Combined spend at freeze: $28.587764.
  Left under the $180 rule: $177.680020.
- Grid estimates from the pilot (expected / x 1.2): Haiku 6 arms, 10 episodes $8.87 / $10.64;
  Sonnet 5 arms, 8 episodes $127.01 / $152.41; Sonnet L4-nogate alone, 8 episodes $13.61 / $16.33.
  Per-run --budget: Haiku $1.25 allowed, $1.55 forced; Sonnet (5 arms) $16.30 allowed, $19.05
  forced (grid.sh defaults; section 8).
- Sonnet sizing (rule below): R = $180 - $2.32 - $10.64 = $167.04. Sonnet 8 episodes x 1.2 =
  $152.41 <= R, so n = 8 (episodes 1-8, halves 1-4 and 5-8), all five core arms. L4-nogate on
  Sonnet (x 1.2 = $16.33) is decided after the core Sonnet grid from the research spend then left;
  at these estimates it would not fit ($167.04 - $152.41 = $14.63), so it runs only if actual
  spend comes in lower.

SONNET SIZING RULE (fixed before freezing, applied mechanically to the pilot's measured costs):
the Sonnet grid always runs all five core arms (B3, B4, B5, L, L4) at all four rho levels and both
conditions; only its number of episodes may shrink. Let R = $180 - research spend at freeze -
the Haiku grid's measured-cost estimate x 1.2. Sonnet runs the largest even number of episodes
n <= 8 (episodes 1..n of questions v4, two halves of n/2) whose measured-cost estimate x 1.2 is
<= R, with a minimum of 6. If even 6 does not fit, Sonnet still runs 6 episodes in the balanced
order and the stop rule halts it where it halts; the episodes completed are analysed as the
secondary family. L4-nogate on Sonnet runs (after the core Sonnet grid, same episodes) only if the
research spend then left covers its estimate x 1.2. The chosen n is recorded below.

## 10. Threats (from ARCH-v4.md section 6, kept short)
- 66 of the 180 corpus items of episodes 1-10 have unperturbed gold values (the core was nearly
  exhausted by v3): a model may know them, and only ungated arms can answer from memory, which
  favours B3/B4 over L4 in test 1. Origin is reported per subgroup.
- Questions are templated and name their target as a backticked qualified name; pre-resolution is
  an upper bound on real-world addressability.
- The C10000 distractors are third-party packages, not more CPython.
- L4 vs L also differs in the episode bound (260 vs v3's 90) and the fixed question folding.
- With 10 (Haiku) and 8 (Sonnet) episodes, equivalence and non-inferiority may often be "not shown".

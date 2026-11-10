# PREREG-v3: context pressure, abstention vs fabrication, and a target-anchored ledger

Status: frozen before the v3 grid. Committed by Mahimn Patel; the commit time is the freeze time,
and the first grid call (calls.jsonl of the first grid run) must be later. Code under test:
L_MECHANISM "l-v3" (commits 0bcfd98, c79ae0c, eeab0d5, dd4ca57). Analysis code: `eval analyze-v3`
(eeab0d5, dd4ca57), run unchanged; anything else is labelled exploratory.

## 1. What came before (reported pilot findings, kept as such)
- v1 (PREREG.md) and pilot v2a/v2b: under context pressure the baselines did not err, they
  abstained. Haiku, W=3000, answerable n=36: B3 rho10 35 correct/0 wrong/1 abstain, rho1000
  28/0/8; B4 31/0/5 and 28/1/7. At W=2000 rho1000, B3 13/0/23 and B4 12/1/23. The planned
  hallucination-rate outcome had almost nothing to measure (a null for H1 as framed in v1).
- Pilot v2c (session facts, Haiku, W=3000, episodes 15-16, answerable n=44 per cell; 8
  unanswerable): B3 rho10 43/0/1 with 3 false answers, rho1000 35/0/9 with 3; B4 32/0/12 with 0,
  33/1/10 with 1; L 21/4/19 with 3, 22/2/20 with 2. L made MORE confident errors than B4 (a
  negative result). All 11 L errors passed Tier 1: 6 cross-wired (a true quote about another
  entity), 3 confusable names, 2 stale notes. Sonnet 5.5 checks (1-6 episodes) were similar.
- These are pilot findings; none of the pilot episodes (v2s seed 1) are reused below.

## 2. Changes since v2 (frozen)
- L's mechanism only (never the grader): the answer gate (verify/gate.py) parses the asked target
  from the question text alone and passes an answer only if a passing citation witnesses that exact
  target (anchored path run, module prefix + whole token, or a note with the exact service token
  and attribute phrase) and holds the answer's value (multi-hop: a second citation through the
  named constant); any failing unsuperseded claim blocks; a cited note updated by a later note
  fails stale_superseded. The 11 v2 failures are regression tests (tests/test_gate_v2_failures.py).
- One grader for all arms: eval/score.py, unchanged since v2 (exact match; answerable -> correct /
  wrong / abstain; unanswerable -> false_answer / correct_abstain).
- A second factor, condition, crossed with arms: `allowed` = prompt clp-v3-allowed (byte-identical
  to clp-v2-pilot-3, abstention allowed) and `forced` = clp-v3-forced ("Always give your best
  answer. Never write UNANSWERABLE ..."), identical across arms.

## 3. Questions (primary research questions)
Q1 (H1). Under context pressure, do models abstain or fabricate, and how does forcing an answer
change that across rho? Q2 (H2, H3). Does the ledger with the fixed verifier reduce confident
errors, especially look-alike-name and stale-fact errors, vs B3 and B4? Q3 (H4). What does it cost?

## 4. Design (the exact grid)
- Corpus ca8e01d67009c (built at W=10000; at W=3000 effective rho is about 3.3x the label).
  W = 3000 for every run.
- Questions: `questions build --version v3 --seed 2 --episodes 19` (sha256
  d6da3013c0640d3d03b0a3d3201a223f0de0d0e69d018a038b958d5235dd5b58): 26 questions per episode
  (20 corpus + 6 session probes), 494 in 19 episodes; 418 answerable, 76 unanswerable; templates
  constant 176, session 114, default 90, locate 61, raises 47, default_ref (2-hop) 6; 161 with
  confusers at some rho, 38 superseded (stale-fact) probes. 19 episodes is the generator's maximum
  on this corpus. Episode 19 was used for the smoke test and is EXCLUDED from all tests.
- Arms B3 (BM25 RAG), B4 (compaction at 0.85 W), L (l-v3). Conditions allowed, forced.
- Primary family, Claude Haiku 5.5: rho in {10, 100, 1000} x condition {allowed, forced} x arms
  {B3, B4, L} x episodes 1-10 (260 questions per arm, rho and condition; 7,800 rows).
- Secondary family, Claude Sonnet 5.5: rho in {10, 1000} x both conditions x 3 arms x episodes 1-4
  (104 questions per cell; 2,496 rows). Reported separately; underpowered by design.
- Run order: Haiku episodes 1-5 over all (rho, condition) runs, then Haiku 6-10, then Sonnet 1-2,
  then Sonnet 3-4, so a halted grid is balanced. Run dirs runs/v3-grid-<model>-<cond>-r<rho>-e<a>-<b>.
  The spend guard allows only one live run at a time, so runs are sequential.
- Why this size: time, not money, binds (the spend lock serialises runs; about 6 min per
  Haiku episode x 3 arms and 10 min per Sonnet one). Expected wall time about 8.5 h.

## 5. Forced condition: behaviour per arm (pre-specified)
- B3, B4: one answer call. If the model still writes UNANSWERABLE, the grader scores an abstention
  and the row is flagged forced_violation.
- L: must still cite and pass Tier 0/1 and the gate. UNANSWERABLE gets one "you must give a best
  guess" message within the round bound. If the bound is hit while the latest candidate is blocked,
  the row's answer is that candidate (reason forced_blocked, forced_fallback=True). If the bound is
  hit and L never produced a candidate (reason bound_hit or budget_exceeded), the row is a
  non-answer (forced_noanswer; scored as an abstention). The smoke run showed this case is common
  for L (Haiku 11/26, Sonnet 9/26), so it is reported per cell.
- PRIMARY scoring of L-forced counts fallback answers as answers; the "gated" variant counts
  forced_fallback rows as abstentions. Unanswerable items stay in: any answer to one is a
  false_answer (fabrication).

## 6. Outcomes
- hr = wrong / answerable. fabrication = (wrong + false_answer) / all items ("confident errors").
  abstain rate = abstain / answerable. Look-alike items: confusers > 0 at the run's rho, plus every
  unanswerable item. Stale items: superseded session probes. Cost: per-question USD from the
  ledger's billed usage.

## 7. Hypotheses and tests (as implemented in analyze-v3; alpha .05, one-sided)
- H1 (forced; B3 and B4 separately): hr rises with log10 rho. Slope by OLS over the per-rho means
  of per-episode rates; p = share of 10,000 episode-cluster bootstrap resamples (seed 0) with
  slope <= 0. Secondary H1sec: the same for fabrication. H1a (allowed): abstain-rate slope > 0.
- H2 (allowed): L has fewer confident errors than B4 (H2-vsB4) and than B3 (H2-vsB3): exact
  McNemar on item-paired error indicators, pooled over rho. Subgroups (secondary): look-alike,
  stale.
- H3 (forced, rho >= 100): L (PRIMARY, fallbacks counted) fabricates less than B4, exact McNemar
  item-paired. Secondary: H3-gated.
- H4 (allowed): median per-question cost L / B4, 95% episode-cluster bootstrap CI (10,000, seed 0);
  supported if the upper bound <= 1.5.
- Holm across {H1-B3, H1-B4, H2-vsB4, H2-vsB3, H3} within the Haiku family; Sonnet is its own
  family. Everything else (H1sec, H1a, subgroups, H3-gated, H4 for Sonnet, any plot by template)
  is secondary or exploratory and labelled so.
- Verdicts: supported / not detected; an effect in the opposite direction is reported as such.

## 8. Stopping rules and budget
- Global hard cap $150; spent before v3 $5.306; smoke test (below) $1.058; so $143.63 left at
  freeze; reserve $20 -> at most $123.63 for the grid.
- Spend estimate from the smoke run: Haiku about $0.016 per episode-arm (3 arms: $0.047-0.050 per
  episode); Sonnet about $0.32 per episode-arm ($0.96 per episode). Haiku 10 ep x 3 rho x 2 cond =
  $3.0; Sonnet 4 ep x 2 rho x 2 cond = $15.4; total about $18.4. Per-run caps: Haiku $0.60 per
  5-episode run (12 runs), Sonnet $4.00 per 2-episode run (8 runs); total caps $39.20.
- Halt (and report) if cumulative v3 grid spend exceeds 1.5x the estimate ($27.60) or the global
  remainder would fall below $20. A run that hits its cap stops; its completed episodes are
  analysed, the incomplete episode is dropped (analyze-v3's superseded/incomplete handling).
- No outcome-based early stopping, no added episodes after looking at outcomes.

## 9. Smoke test (disclosed, before this freeze)
- On Nov 1, 2026, 13:19-13:40 EST, episode 19 only, rho 1000: Haiku allowed (3 arms, $0.047),
  Haiku forced ($0.050), Sonnet forced ($0.961); $1.058 total. Purpose: cost per unit, run time,
  and whether forcing works. Observed (not used in any test): Haiku-forced B3 still abstained on 9
  of 26 (forced_violation); B4 complied fully (4/4 false answers on unanswerable); L hit its round
  bound without a candidate on 11 of 26. This led to section 5's forced_noanswer definition and the
  grid size; no hypothesis or test was changed after seeing it.

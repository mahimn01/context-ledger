# Pre-registration: context-ledger pilot (M4)

Frozen 2026-10-31 (America/Toronto), before any grid call. The commit that adds this file is
the freeze point. Changes after it are allowed only as dated amendments in section 10,
committed before the first grid call. No amendment may be made after any grid result has been
seen. Any analysis not listed here is labelled **exploratory** wherever it is reported.

Author: Mahimn Patel.

## 1. Question

Does hallucination rise as the material grows relative to the context window, and does a
context ledger (hash-verified reads, span citations, a tiered claim verifier and
payoff-gated folding) flatten that rise, at a bounded cost?

## 2. Design

**Material.**
- CPython 3.15.0 source (the verified tarball), perturbed deterministically (seed 1).
  Constants and default values are changed, and identifiers are renamed, so that
  memorization cannot answer.
- Three nested corpora, C10 ⊂ C100 ⊂ C1000. The C-ρ corpus holds about ρ·W tokens, counted
  with the calib-v1 estimator for `claude-haiku-5-5`, which was fitted live against
  `count_tokens` on 2026-10-31.

**Ratios.** ρ ∈ {10, 100, 1000}. ρ is the material-to-window ratio.

**Window.**
- W = 6,000 tokens. W counts the system prompt and the conversation; the fixed tool
  definitions are not counted.
- W is the same for every arm, model and ratio.
- Why 6,000: in the 2026-10-31 live smoke runs (Haiku 5.5, ρ=10), the ledger arm grew by
  about 2,000 tokens per question without pressure. At W=16,000 no fold and no compaction
  fired. At W=6,000 both fired.
- Corpora are rebuilt at W=6,000, so the ratio holds exactly as defined.

**Questions.**
- 12 episodes of 8 questions, seed 1, generated from the perturbed gold files, all of which
  are in C10.
- Each episode has 6 answerable questions (3 anchors and 3 fold probes on the anchors' spans)
  and 2 unanswerable questions (near-miss symbols absent from every corpus).
- Totals: 96 questions per cell (72 answerable, of which 36 are probes, and 24
  unanswerable).
- The same questions are used in every arm, ratio and model, so the comparisons are paired.

**Arms** (as defined in `src/context_ledger/eval/arms.py`; same model, W, `max_tokens` of
1,024, answer format and system framing):
- **B3, RAG.** One shot per question; BM25 top-k chunks fill W.
- **B4, agent with LLM compaction.** Corpus search and read tools. At 85% of W, the
  conversation is summarized by the same model.
- **L, ledger hybrid.** The M2 LedgerClient/Session with the corpus tools and the ctx tools.
  It uses the Tier 0/1 verifier on egress and a payoff-gated backstop at a hard 85% and a
  target 50% of W.

**Models and cells.**
- `claude-haiku-5-5` with thinking disabled: B3, B4 and L × ρ ∈ {10, 100, 1000} × 12
  episodes (the "Haiku grid").
- `claude-sonnet-5-5` with `between_tools` thinking: B4 and L × ρ ∈ {10, 1000}, with the
  episode count set by the planner within the sub-cap, at most 12 (the "Sonnet subset").
- Sampling parameters cannot be set on these models: temperature, top_p and top_k return
  400. Each cell is run once.

**Budget.**
- The hard cap is $50 (SpendGuard).
- Planned sub-caps: re-smoke $0.75, Haiku grid $6.00, Sonnet subset $33.00. The total
  planned spend is therefore at most $40, with at least a 20% reserve.
- If a sub-cap or the cap stops a run, the completed episodes are analyzed and the shortfall
  is reported.

## 3. Grading (arm-independent)

**The grader is not the ledger.** The Tier 0/1 verifier is part of arm L's mechanism. It
never decides an outcome.

**Correctness.**
- Every arm's final answer is compared with the gold answer by the same exact-match rules
  (`eval/score.py`): literals by value; module paths normalized; exception sets by set F1 = 1.
- Outcomes:
  - answerable questions: `correct`, `wrong` or `abstain`;
  - unanswerable questions: `correct_abstain` or `false_answer`.
- Hitting the call bound counts as an abstention (flagged as `bound_hit`).

**Citations.**
- Every arm's citations are exported in one format (`path:start-end` plus a verbatim quote).
- They are checked by the same function, against the perturbed file text at the cited lines.

**Hallucination type.** This is descriptive only.
- A `wrong` or `false_answer` outcome is `unsupported` if any citation fails (or there are
  none).
- Otherwise it is `misread`.

**Caught (L only, descriptive).** An answer that L's egress check blocked, that the grader
would score wrong, and that is followed by a correct answer or an abstention.

**Leakage.** An answer equal to the unperturbed upstream value.

## 4. Metrics (per arm × ρ × model; 95% cluster-bootstrap CIs over episodes, 10,000 resamples, fixed seed)

**Primary.**
- **HR**, the hallucination rate on answerable questions: wrong / answerable.

**Secondary.**
- False-answer rate: false_answer / unanswerable.
- Total hallucination: (wrong + false_answer) / all.
- Abstention: abstain / answerable (abstention on unanswerable questions is 1 − false-answer rate).
- Accuracy: correct / answerable.
- Selective accuracy: correct / answered answerable.
- Fold-probe accuracy.
- Leakage rate.
- Cost per episode (USD, settled by SpendGuard), cache-hit share, calls per episode.
- Folds and compactions per episode, and predicted vs actual fold savings.

## 5. Hypotheses and tests

The **slope** is the ordinary least-squares slope of the cell rate on log10 ρ (a linear
probability model). It is always estimable, including when cells have zero events. Its 95% CI
and one-sided p-value come from the cluster bootstrap over episodes; the one-sided p-value is
the fraction of resamples with the slope on the wrong side of 0. A logistic slope is reported
as secondary where it is estimable.

**H1.** Under RAG and LLM compaction, hallucination rises with log ρ.
- Test: for B3 and for B4 (Haiku), the HR slope is > 0, one-sided, α = 0.05.

**H2.** The ledger flattens that curve.
- Test: on Haiku, slope(L) − slope(B4) < 0, one-sided, with the bootstrap over episodes
  paired across arms.
- Supporting test: an exact McNemar test (two-sided) on paired per-question `wrong` vs
  not-`wrong`, L vs B4, at ρ = 1000.
- Falsifier: if L's slope is not below B4's, H2 fails whatever else holds.

**H3.** Hybrid folding beats model-only folding. This is only partly testable here: the pilot
has no harness-only fold arm, so that half of H3 is not tested.
- Test (Haiku, ρ ≥ 100 pooled): HR(L) < HR(B4), one-sided bootstrap.
- Non-inferiority: the 95% CI lower bound of accuracy(L) − accuracy(B4) must be > −3
  percentage points.

**H4.** Cost overhead is bounded.
- Test: the ratio of median episode cost, L / B4, on Haiku pooled over ρ. The 95% bootstrap
  CI upper bound must be ≤ 1.5.

**Multiplicity.** Holm correction over the five primary Haiku tests: H1-B3, H1-B4, H2 slope,
H3 and H4 (H4 judged by its CI). Sonnet results are a secondary replication: the H2
difference and McNemar test at ρ = 1000 and the H4 ratio, reported uncorrected.

**Power caveat.** With 12 episodes and 72 answerable questions per cell, only large effects
are detectable. Null results are reported as "not detected at this n", not as "no effect".

## 6. Exclusions

- **Infrastructure errors.** A question whose call failed for an infrastructure reason (an
  API 5xx, a timeout or a harness exception, but not a model behaviour) is excluded, and
  every such exclusion is counted and listed.
  - An episode with an infrastructure failure may be re-run once, identically, before
    analysis. Re-runs are logged.
  - No episode is re-run for any other reason.
- **Pairing.** Paired tests use only episodes complete in both arms of the pair.
- **Not excluded.** Leaked answers (flagged; a sensitivity analysis without them is
  exploratory), bound hits (counted as abstentions) and budget-stopped episodes (they are
  incomplete and analyzed as far as they went, and the shortfall is reported).

## 7. Analysis outputs

`eval analyze` writes:
- per-cell metrics as CSV;
- the hypothesis tests as CSV and JSON;
- the main figure: HR (with 95% CIs) against log10 ρ, one line per arm, for each model.

## 8. Deviations from the full design (on the design Page), stated in advance

- One domain (Python), not two.
- Three ratios, not five.
- Three arms, not eleven.
- 12 episodes, not 40.
- No Tier 2 (NLI) verifier.
- BM25 retrieval only.
- Fixed-effects linear-probability slopes with a cluster bootstrap instead of a mixed-effects
  logistic regression (n = 12 clusters; stdlib-only harness).
- No sampling-temperature control.

## 9. Integrity rules

- Scoring code and this file are frozen at the freeze commit. Changing the scoring after
  seeing grid results is forbidden; any such analysis is exploratory and is labelled so.
- All raw artifacts (results, calls, ledgers, spend ledger) are kept.

## 10. Amendments

### Amendment 1 — 2026-10-31, 10:55 PM EDT. Made before any grid call; no grid result existed.

The M4 fixes were built after the freeze. None of them changes the hypotheses, the metrics,
the grader or the tests in sections 3–5. They do settle details that section 2 left open, so
they are recorded here.

1. **How W is measured.**
   - Every arm measures W with the `calib-v2` estimator, which is the calib-v1 fitted slope
     applied to each piece, with no fixed per-text intercept.
   - Reason: calib-v1 added its 190-token fixed term to every message segment. That inflated
     L's measured window about 2.4× and B4's about 1.6× against true counts, so the arms ran
     under different effective windows. This violated "W is the same for every arm".
   - Corpus sizes still use calib-v1, as section 2 says. SpendGuard keeps the conservative
     calib-v1 bound.

2. **Corpus and questions** (corpus `c7580b09b2432`, seed 1):
   - C10 holds 60,703 tokens in 18 files, C100 600,301 tokens in 91 files, and C1000
     6,019,300 tokens in 571 files.
   - C10 files are ordered deterministically by question-candidate density. The density is
     computed on the unperturbed source with the templates and filters only; no answer, gold
     value or model output is used.
   - A gold file may serve two episodes, with no question repeated: 18 gold files, 6 of them
     shared.
   - "Locate" questions are paired only with each other, so no earlier question text (such as
     `mod.f`) gives away a locate answer. Before this rule, 9 of 17 locate questions had leaked
     this way.
   - Result: 96 questions in 12 episodes, 72 answerable (36 fold probes) and 24 unanswerable.
     The template mix is constant 32, default 25, raises 9, locate 6.

3. **Tier 1 witness rule** (arm L's mechanism only, never the grader):
   - An identifier in a claim is witnessed by the quote or by a contiguous run of whole
     components of the anchored path (module and qualname) of a passing citation.
   - If the quote occurs exactly once in a cited class span, the method containing it also
     witnesses.
   - Values (numbers, strings, exact tokens) must still appear in the verbatim quote.
     Backticked numbers are compared by value, and backticked strings by their content.
   - Known, unchanged: an L answer with an empty claims list passes egress. This is
     pre-existing arm behaviour and is reported as a threat to validity.

4. **Answer format.** All arms use prompt `clp-m4-pilot-v2`: a plain `ANSWER:` line, with
   claims that state the subject and the value.

5. **Analysis details fixed in `eval analyze`:**
   - H4's p-value is the share of bootstrap resamples with a ratio ≥ 1.5.
   - H3 counts as supported only if its non-inferiority check also holds.
   - The McNemar test is supporting evidence and sits outside the Holm family.
   - The secondary logistic slope uses a cluster-robust Wald interval.
   - Episode-cost figures use complete episodes only.
   - If a unit appears in more than one run directory, the run listed last wins. The analysis
     lists each grid directory once.

6. **Smoke episodes.** The re-smoke runs episodes 1–2 at ρ 10 and 1000, outside the grid.
   Smoke results are not analyzed. The grid re-runs every episode from scratch.

7. **Bug fix in arm L found by the re-smoke: the current question is never folded.**
   - In re-smoke 1 (ρ = 1000, L, episode 2) a backstop fold took the current question's user
     segment as a victim. The model then answered an earlier question from the fold summary.
   - Fix (commit `55ac0a4`): the first user segment since the last `answer_emitted` cannot be
     a fold victim, for backstop folds and gated model folds alike. This matches B4, whose
     compaction always keeps the current question. Earlier, answered questions can still be
     folded.
   - Re-smoke 1 L wrong answers: one (ρ = 1000) came from this bug. The other (ρ = 10) was
     not: the question stayed in the window and the model re-answered the previous question
     with a grounded claim. That is model behaviour and will be scored as a wrong answer.

8. **Known gap, unchanged:** Tier 1 checks that a claim is grounded, not that it answers the
   question asked. A grounded answer to a different question passes egress and is scored as
   wrong by the arm-independent grader. This is reported as a threat to validity.

9. **Re-smoke.** Re-smoke 1 is `runs/smoke-m4r1-*`. Re-smoke 2, with the fix in item 7, is
   `runs/smoke-m4r2-*` (run after `55ac0a4`; go criteria met: L 21 correct, 0 wrong, 3 abstain on 24 answerable; 5 blocked answers; 6 L folds, 3 B4 compactions; spend $0.050, leakscan pass). Both are Haiku, W = 6000, episodes 1–2, 8 questions each, at ρ 10 and
   1000. Neither is analyzed.


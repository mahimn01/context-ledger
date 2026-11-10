# PREREG-v3 Amendment 1: L's empty-text cache_control bug, its fix, and the rerun of the missing L units

Committed before any rerun call. Everything not stated here is as frozen in PREREG-v3 (e803fed).

## What happened
During the v3 Haiku grid (episodes 6-10), three runs stopped early with `incomplete: true`,
`stopped: {arm: L, reason: api_errors}`: the API returned HTTP 400 "cache_control cannot be set
for empty text blocks" on L requests. B3 and B4 completed in those runs. The grid log reported
these runs as exit 0 because `$?` was grep's status in a pipe, not the run's.

Affected units (model Haiku 5.5, W=3000, arm L only):
- runs/v3-grid-haiku-allowed-r10-e6-10: L episode 6 partial (23/26 rows), episodes 7-10 missing.
- runs/v3-grid-haiku-forced-r10-e6-10: L episode 6 complete, episode 7 partial (5 rows), 8-10 missing.
- runs/v3-grid-haiku-forced-r100-e6-10: L episode 6 complete, episode 7 partial (15 rows), 8-10 missing.

## Fix
Commits 4d7c711 and 47abe39. Root cause: L's verification message to the model was empty when an
earlier claim in the turn still needed a refetch but no claim in the current answer did
(`_refetch_message` wrote lines only for current claims), and the request builder put a cache
breakpoint on that empty block; the API refused it, and L's retries hit the same request until
the run stopped. The fix (a) makes that verification message name the earlier claim and what to
refetch (new wording, used ONLY in the case that previously crashed; no completed L unit in the
81 v3 L ledgers ever reached it, so all completed units are unaffected), (b) makes the request
builder drop empty or whitespace-only text blocks (an otherwise empty message sends "(empty)",
as B4 already does) and move a breakpoint aimed at a dropped block to the nearest earlier block,
byte-identical otherwise, and (c) makes the fake server refuse empty text blocks with the API's
400. The three stopped requests, rebuilt read-only from their ledgers, reproduce the API's error
with the old builder and are valid with the fix (tests/v3fix_stopped_windows.py). L's mechanism
label stays `l-v3`; rerun L rows record `harness_patch: "v3.1-empty-text-cache-control"`. The fix
changes no prompt template, tool, gate, question, corpus, window, model, bound or budget. Item (a)
is a deviation from the frozen mechanism, disclosed here: it changes what L's model sees in the
one situation that previously made the run fail. B4 has a latent form of the same bug (an empty
model reply plus a tool call would be replayed as an empty block); no B4 transcript has one and
B4 is unchanged.

## Rerun (L only)
Run ids runs/v3-grid-haiku-<cond>-r<rho>-e<a>-<b>-Lrerun, `--arms L`, same questions v3 (same
seed and file sha256), corpus ca8e01d67009c, W=3000, model, condition prompts and per-question
bounds:
- allowed rho10 episodes 6-10; forced rho10 episodes 7-10; forced rho100 episodes 7-10.
Per-run cap $0.60. Exit status is taken from the run itself (zsh pipestatus) and any run.json
with `incomplete: true` is reported as INCOMPLETE. If a rerun unit fails again it is reported as
missing, not retried with any other change.

Funding (no scientific effect): per Mahimn's instruction of Nov 1, 2026, 11:51 PM EST, new paid
calls use a second Anthropic key funded for this research, with its own spend ledger
(commit 8efcef9; `--key research`, ledger .clp-spend/research/spend.jsonl, hard max $200, $20 reserve). Same model ids and price table.

## Analysis
analyze-v3 merges runs by unit (model, window, condition, arm, rho, episode): incomplete units are
dropped and listed in `dropped_units`; when a unit is complete in more than one run the later run
wins and the earlier is listed in `superseded_units`. In the grid as run, the complete L episode 6
units of the two forced runs are kept from the original runs (only the missing units are rerun),
and the partial units are dropped and replaced by the rerun. All H1-H4 tests, Holm correction and
the figure are recomputed on the merged data, and both the pre-fix (incomplete) and the merged
verdicts are reported. Nothing else changes.

# ARCH-v4: L4 ("l-v4"), an addressable, verified working set (EXPLORATORY)

Status, 2026-11-02: built and tested offline on branch `v4` (fake model only; no live call was
made). Nothing here is a result about any model. The analysis is exploratory until a PREREG-v4
freezes it. v3 is unchanged and byte-for-byte reproducible (tests/test_v3_reproducible.py).

## 1. Why: what L-v3 did in the v3 Haiku grid

Read-only diagnostic of the v3 Haiku grid (W = 3,000, questions v3, corpus ca8e01d67009c; L units
as analyze-v3 loads them; script and full tables kept outside the tree). Three findings change
the picture of L-v3's low accuracy more than the reason counts do.

**1. Most `bound_hit` rows are questions that were never asked.** The episode bound (90 model
calls) ran out in 29 of 30 allowed and 22 of 22 forced L episodes. A v3 episode has 26 questions;
L used 4.1 (allowed) and 5.7 (forced) calls per question it was asked.

| condition | L rows | bound_hit | never asked (0 calls) | episode bound mid-turn | 10-call bound |
|---|---|---|---|---|---|
| allowed | 780 | 177 | 126 (16% of rows) | 17 | 34 |
| forced | 572 | 239 | 226 (40% of rows) | 5 | 8 |

**2. The question text was folded out of the window in turns that carried a note.** The backstop
protects "the first user segment since the last answer" as the current question; L appended the
note span before the question text, so the note was protected and the question folded.

| condition | turns with a note: question still verbatim at the last call | turns without a note |
|---|---|---|
| allowed | 145/223 (65%) | 431/431 (100%) |
| forced | 71/123 (58%) | 223/223 (100%) |

Of the 16 allowed model_abstained corpus rows whose gold was verbatim at the last call, only 2
still had the question in the window.

**3. L mostly did not find the asked text, and never re-found a folded note.**

| what L did | allowed | forced |
|---|---|---|
| bound_hit with a turn (n) | 51 | 13 |
| model calls per row (mean) | 7.6 | 7.2 |
| corpus_search / corpus_read / ctx_refetch per row | 4.7 / 3.8 / 0.2 | 4.8 / 3.0 / 0.0 |
| repeated identical corpus_read | 27 of 196 | 5 of 39 |
| gold file opened by turn end (answerable corpus) | 44/46 (96%) | 11/11 |
| gold span appended this turn | 18/46 (39%) | 3/11 (27%) |
| gold verbatim at the last call | 5/46 (11%) | 1/11 (9%) |

| model_abstained, answerable (allowed, rho pooled) | n | gold verbatim at last call | gold summary_only | unseen (file read, gold lines not) | file never opened |
|---|---|---|---|---|---|
| corpus questions | 79 | 16 | 2 | 27 | 34 |
| session probes (latest note) | 56 | 4 | 52 | 0 | 0 |

Locate questions: the defining file was opened in 3 of 34 abstentions. Session probes: the latest
note was folded (summary_only) in 52 of 56, and the refetch-before-abstaining reminder never
fired on a session probe: `tools.question_identifiers` finds no identifier in "What is the current
auth scope of service mintaka-gateway?".

Forced, the 113 forced_blocked fallbacks were blocked by the gate's target_not_witnessed in 70
(the gold was verbatim at the last call in only 9 of those 32 corpus rows), by Tier 1 reasons in
37, by stale_superseded in 3 and value_not_with_target in 3; 79 of 113 used all 10 calls.

So L-v3's accuracy gap is mostly navigation (it searches, reads 750-token windows of the wrong
part of the right file, loses the question or the note to folding) and the episode bound, not the
gate. v4 attacks exactly that: an exact address for the asked name, read before the first call;
a gate refusal turned into a targeted read of that address; the question protected.

## 2. Design

### 2.1 Address index (src/context_ledger/index/)

Built once per corpus by `eval index build` (no model call) from the corpus file texts only; the
builder's signature takes texts, each file's smallest rho, the rho list and the output path, so no
question, gold or run field can reach it. One SQLite file per corpus, `address-v4.sqlite` in the
corpus cache directory, with tables per rho level r holding only C-r's files:

- `sym_r`: every addressable name, sorted by (parent, leaf): modules, classes, defs (not inside a
  function), module constants (simple Name targets of module-level assignments, the anchors L
  registers) and defaulted parameters. Qualified names follow `anchor.python` (module parts of
  the path, then the qualname), the names Tier 1 and the gate anchor by. Each row carries its
  address: path, 1-based inclusive lines (decorators included) and the sha256 of those lines.
- `leaf_r`, `rleaf_r`: the same names by last component and by reversed last component.
- `node_r`: the Merkle tree of directories and files. A file's hash is the sha256 of its bytes; a
  directory's is the sha256 of its children's sorted "kind name hash" lines; the root hash of
  each level is recorded in every v4 run's run.json, so a run is tied to the exact bytes of C-rho.
- Summaries are deterministic one-liners (a def's signature with defaults elided, "class X(bases)",
  "NAME = …", a module's first docstring line) and never show a value: values are only ever read.

A lookup is a binary search over ranks, one SQLite rowid probe per step, counted ("visited"):
exact names cost ceil(log2(N+1)) probes; a miss walks the name's prefix levels (one search each),
scans the siblings of the longest existing prefix (at most 2,048 rows) for names within two edits,
and reads a few neighbours in the last-component and reversed orders. Every near miss is labelled
"different identifier" and why (1 edit under `xml.dom`, the same name in another module, a shared
ending). Root children (every top-level module) are never scanned.

The session-note table (`index/notes.py`) maps (service, attribute) to the notes stating it, in
the order shown; the latest wins. It is built from the note text the user showed, with the gate's
own "the <attribute> of service <service>" parse, so its supersession rule is the gate's.

### 2.2 Tools (L4; B5 has lookup, outline, read)

`lookup(name)`, `read(address)` (hash-checked, then corpus_read's cover and cap: python spans of
the symbols the lines cover, text spans otherwise, each counted with its alias header, min(2,000,
W/4) tokens), `outline(target)`, and `search(query)` (BM25, at most 2 per question; a third gets
`error: search_limit`). In L4 a read registers spans in the ledger exactly as L's corpus_read does,
so Tier 0/1, the gate, folding and ctx_refetch treat them like any span; an evicted or folded span
keeps its anchor, so ctx_refetch or read of its address is exact (the window is a cache over the
content-addressed store). The four ctx tools are unchanged.

### 2.3 The L4 turn (anthropic/client_v4.py, eval/arms_v4.py)

1. The question text is the turn's first user segment, so the backstop protects it (finding 2).
2. The note, if any, as a verbatim span (as in L), then **pre-resolution**: the harness parses
   the asked target from the question TEXT with `verify.gate.asked_target` (the gate's own parser)
   and resolves it: an exact code address is read into the window (whole within W/2, below) with an
   "Index: `X` is at path:a-b#sha, shown as @7" line; a session target gets the latest note of its
   chain, refetched if folded; a miss gets the **address-miss certificate** ("no exact address for
   `X`; nearest DIFFERENT identifiers: …"), which the ledger records verbatim as a user segment.
   `pre_resolve(question_text, address, notes)` takes a string, never the question dict.
3. The v3 loop: Tier 0/1 per claim, the answer gate, refetch routes, fix messages, the
   refetch-before-abstaining reminder, max 10 calls per question and 90 per episode.
4. **Repair**: on a gate refusal (target_not_witnessed, value_not_with_target, stale_superseded)
   the harness holds the asked target's anchored span verbatim (append, or refetch if folded; a
   verbatim span is only pointed to) or the latest note of the chain, and sends "Repair (<reason>):
   why, where the text is, cite it". At most 2 repairs per question, counted by reason
   (`repairs`, `repair_reasons`), and only while a round is left to answer in; later refusals get
   v3's gate message.
5. **Forced**: an UNANSWERABLE gets v3's one best-guess message. Before the last round, if a
   candidate is blocked, the harness says it is the last round; if the previous refusal was not
   just repaired and a repair is left, it repairs once more first ("final" in repair_reasons;
   "final_overflow" if the backstop then cannot make room for the call): the fallback goes
   through repair and the gate like any answer. If the turn still ends blocked, the row's answer
   is the last candidate, flagged `unverified_guess` (and `forced_fallback`; reason
   unverified_guess); the ledger records an abstention, so nothing unverified leaves the egress.
   No candidate, whatever ended the turn: reason `forced_noanswer`. `stop_reason` keeps v3's
   reason (bound_hit, model_abstained, budget_exceeded, ...) so L4 and L rows stay comparable.

**Refinement (the harness's own reads).** Pre-resolution and repair read the target's anchored
Python span whole when it fits W/2 tokens (1,500 at W = 3,000), else its first lines under
corpus_read's cap (W/4, 750). The reason surfaced in an offline run on the real corpus: a def
longer than the read cap comes as a text span of its first lines, and neither Tier 1 nor the gate
can anchor `lzma.LZMAFile.__init__` to a text span (they witness a qualified name by the cited
span's own anchored path, or a quote that spells it), so the answer failed Tier 1
(identifier_not_anchored) before the gate could refuse and repair it. Every answerable code target
of questions v3 and v4 fits W/2 (the largest is 1,424 tokens; 11 of v3's 342 and 12 of v4's 342
exceed 750). Reads the model makes keep the W/4 cap, as in v3, and B5's pre-resolution uses the
same W/2 rule. l-v3's corpus_read has the same text-span limitation on long defs, which is part
of its identifier_not_anchored blocks; Tier 1 itself is unchanged (P5 re-verifies with it).

### 2.4 B5, the control

B3 plus the same pre-resolution and lookup/outline/read, with NO gate and NO repair: a fresh
conversation per question, the first message holding B3's BM25 context (chunks and the episode's
earlier notes), the pre-resolved text as a read block or note with its address (or the
certificate), and the question; up to 10 calls per question and 90 per episode (as B4, L, L4);
the prompt held within W - max_tokens (B3's rule) by dropping the lowest-ranked chunk, then
eliding the oldest tool result to a stub, else abstaining with budget_exceeded. Under that rule
B5 reads the pre-resolved span whole only when the first request still fits (a target over the
read cap may come as its first lines; B5 has no Tier 1 to anchor). B5 vs B3 isolates the
addressable index; L4 vs B5 adds the ledger, the gate and repair (not the gate alone: see threats).

### 2.4b L4-nogate, the gate ablation (BUILD-V4B decision 3)

L4 exactly, with the answer gate off: `emit_answer` is not given the question, so the gate never
runs and no gate-triggered repair can happen. Tier 0/1 grounding, pre-resolution, the tools, L4's
prompts (same sha256) and the forced handling are L4's: a candidate Tier 0/1 still blocks at the
end of a forced turn is emitted as `unverified_guess`, after the same "final" repair. Rows carry
`gate: "off"` (L4's `"on"`) and mechanism `l-v4-nogate`; ledgers go to runs/<id>/L4-nogate/. On
the fake model (tests/test_l4_nogate.py) it sends L4's requests byte for byte and writes L4's rows
(gate fields aside) wherever the gate passes, and emits where L4 repairs. L4 vs L4-nogate is the
gate's effect; L4-nogate vs B5 is the ledger's (per-episode conversation, folding, ctx tools, Tier
0/1).

### 2.5 Prompts, labels and run.json

`clp-v4-allowed` / `clp-v4-forced` are clp-v3-allowed / clp-v3-forced with tool paragraphs for L4
and B5 added; B3, B4 and L get v3's exact prompts and sha256s. A run with B5 or L4 uses them and
adds a `v4` block to run.json (mechanisms l-v4 and b5-v4, the index version and the Merkle root of
C-rho, the repair and search bounds, the pre-resolution cap); a run of B3, B4 and L alone writes exactly v3's
configuration (tested). `--max-calls-episode` (default 90) is recorded only when changed. v4
rows carry `preresolved`, `repairs`, `repair_reasons`, `unverified_guess`, `tool_counts` and
`mechanism`; v3 rows are unchanged.

### 2.6 The extension level (eval/extend.py) and questions v4

`corpus extend --base ca8e01d67009c --rho 10000` (from the source snapshot, section 6) builds corpus **cf9bb5aff4ed6**: C10, C100 and C1000
are the base's files byte for byte (same chunk rows, same BM25 ranking, tested), and C10000 adds
open-source Python packages installed on this Mac as distractors only. The frozen source list
`src/context_ledger/eval/ext_sources_v4.json` (tracked) names 112 distributions with permissive
licenses (MIT, BSD, Apache, PSF, ISC, MPL, MIT-CMU; GPL/LGPL and unknown excluded), one copy each
(the newest Python's), with version, license and a digest over every eligible file; the build
re-reads and refuses if any digest differs. Files keep their site-packages paths under
`Lib/site-packages/` (where CPython's tree has third-party packages), in a seeded order after the
base's, until W x rho; .py files get the base builder's literal perturbation (constants and
defaults; never its renames); every file's sha256 and source sha256 is in the manifest.

| level | files | bytes | tokens (calib-v1, conservative) | effective rho at W = 3,000 | tokens (calib-v1 point estimate) | effective rho |
|---|---|---|---|---|---|---|
| C10 | 25 | 151,096 | 101,372 | 34 | 60,430 | 20 |
| C100 | 114 | 1,662,261 | 1,086,883 | 362 | 634,110 | 211 |
| C1000 | 856 | 15,391,456 | 10,029,977 | 3,343 | 5,831,889 | 1,944 |
| C10000 | 9,281 | 153,202,644 | 100,003,667 | 33,335 | 58,207,909 | 19,403 |

rho 10000 is reached (8,425 extension files from 110 distributions add 89,973,690 tokens; build
79 s). The levels span three orders of magnitude.

Questions v4 = the v3 generator labelled v4 (ids v4eNNqMM, questions-v4.jsonl), seed 3, built on
cf9bb5aff4ed6: confusers counted at all four levels, near-miss names and session values absent from
all 9,281 files. A run on the extension with a questions version it lacks (v3, for the pilot)
reads the base's file.

**Fresh items (BUILD-V4B decision 2).** Seed 3 alone reshuffled the same pool: 227 of its 494
items overlapped questions v3, and every episode had some. v3's 19 episodes had used almost every
candidate the rho-10 core offers (342 of about 380). So `questions build --version v4` filters
before drawing: an item that overlaps questions v3 episodes 1-10 (the v3 grid), 11-12 (the v4
pilot) or 19 (the v3 smoke) never enters a pool (overlap = same target identifiers and gold
answer, or the same question text lower-cased with whitespace collapsed); a near-miss
unanswerable that overlaps is drawn again; session probes ask only about service/attribute pairs
no reference probe asked. That alone left 6 episodes, under the minimum of 10, so the pools got
extras from every .py file of C10 (so each stays answerable at every level), only for names no v3
item asks about in any episode, in the v3 templates' exact wording (the gate parses them, the
grader grades them unchanged): `constant` for upper-case module constants the perturbation did
not rewrite (octal or hex ints, 0/1 modes, private `_X` names; gold the source literal, no upstream
answer) and `locate` for unrenamed top-level functions in files with renames. Extras come after
the v3 generator's candidates in each file, and a file with no single left may give a fold-probe
pair's anchor as a single (v4 only). Result, seed 3, `--episodes 13`: **13 fresh episodes** (Haiku
uses 1-10, Sonnet 1-8), 338 questions, 286 answerable (234 corpus, 52 session), 52 unanswerable,
0 items overlapping the reference (`questions overlap`); corpus items by origin: v3 generator 137,
unperturbed constants 92, unrenamed locate 5; 101 of the generator's items were also in v3
episodes 13-18, which no run asked. Regenerating questions v3 from the same code gives the
main tree's file byte for byte (sha256 d6da3013c064).

## 3. Invariants

- No arm reads a question's gold or metadata: pre-resolution and the gate get the question text
  and user-visible note text only (sentinel test over every request and every artifact of L4 and B5).
- The index is built from corpus texts only; every name resolves exactly to its own anchored span
  and hash, and one-character edits never resolve to the original (property test).
- Lookups are exact; near misses are labelled as different identifiers, never offered as the one asked.
- Every read is hash-checked against the address it names; a stale address is refused.
- L4's ledgers satisfy P1-P16 (`check` clean in the tests); every emitted answer passed Tier 0/1
  and the gate; unverified guesses exist only in the results row, never in `answer_emitted`.
- The question is the turn's protected current question in L4.
- B3, B4 and L (l-v3) outputs are byte-identical to the pre-v4 tree on a fake-model fixture in
  both conditions (results, calls, B4 transcripts, L ledger heads, run.json configuration).
- The verifier and gate are part of L4's mechanism; the grader (score.py) is unchanged and grades
  every arm alike, unverified guesses as answers.

## 4. Cost model: why L4's per-question cost is O(log N) in corpus size

A question's model cost is bounded by W per call (the backstop and B5's fitting hold every prompt
to W) times at most 10 calls; nothing the model sees grows with the corpus. The harness work per
question is: pre-resolution and each lookup, a few binary searches over the level's sorted names,
O(d log N) probes for a name of d components; each read, O(read cap) bytes; each outline, at most
60 rows; the repair, one more lookup and read. The only corpus-proportional operation is BM25
`search`, capped at 2 per question (B3's retrieval is BM25 by design). Measured on cf9bb5aff4ed6
(380 questions v4 targets per level; this Mac):

| level | names N | exact lookup probes (mean / max) | miss probes (mean) | rows read after search (max: hit / miss) | lookup median / p95 | BM25 search median (top 200) |
|---|---|---|---|---|---|---|
| C10 | 903 | 11.3 / 20 | 57.2 | 17 / 312 | 0.08 / 0.94 ms | 0.7 ms |
| C100 | 4,573 | 14.1 / 25 | 71.4 | 17 / 312 | 0.10 / 1.14 ms | 2.9 ms |
| C1000 | 35,516 | 17.4 / 31 | 89.3 | 17 / 312 | 0.12 / 1.25 ms | 11.3 ms |
| C10000 | 300,955 | 20.9 / 37 | 107.2 | 17 / 312 | 0.14 / 1.37 ms | 85.2 ms |

N grows 333x; exact probes grow by 9.6 (log2 333 = 8.4), rows read stay capped, time 1.7x; BM25
grows about 120x. All 342 answerable code targets resolve exactly to an address containing their
gold line at every level; none of the unanswerable near misses resolves. tests/test_index_v4.py
asserts the probe bounds (ceil(log2(N+1)) for exact hits, at most 6 searches for a miss) and that
time follows the probes on nested synthetic levels of 10, 100 and 1000 modules.

Money (`eval plan --v4`, BUILD-V4B decisions 1 and 7: every arm at 260 calls per episode; costs
from the v3 grid at W = 3,000, which do not grow with rho at fixed W, each agent arm scaled from
v3's 90-call bound by its measured calls per question x 26 / 90; L4 and L4-nogate planned at L's
cost, B5 at twice B3's; Sonnet's v3 costs are about 20x Haiku's, measured per arm). The pilot
replaces the estimates (`plan --v4 --measured <pilot runs>`).

| item | expected | x 1.2 |
|---|---|---|
| pilot: 6 arms, Haiku, questions v3 episodes 11-12, rho 10/1000/10000, both conditions | $1.66 | $2.00 |
| grid Haiku: 6 arms, 10 episodes of questions v4, rho 10/100/1000/10000, both conditions | $11.09 | $13.30 |
| grid Sonnet: 5 arms (no L4-nogate), 8 episodes, the same rho and conditions | $141.17 | $169.40 |
| grid Sonnet: L4-nogate alone, 8 episodes | $33.70 | $40.44 |

The Sonnet grid as decided does not fit the research key's usable $180 together with the Haiku
grid and the pilot (before counting what is already spent); the scripts' stop rule would halt it
part way. PREREG-v4 leaves that choice open (fewer Sonnet episodes or rho, or fewer arms).

## 5. Novelty, against prior art

Most of v4 is not new, and this document does not claim otherwise. Paging a bounded context
over an external store is MemGPT/Letta's virtual context; a hierarchical summary tree is RAPTOR's
(ours is deterministic, for navigation and integrity, not retrieval); architectural long-term
memory (Infini-attention, the Recurrent Memory Transformer, LongMem) needs training and is
orthogonal to an inference-time harness; gist memory with page lookup is ReadAgent's, and our
fold-then-refetch is the same move; MemAgent learns an overwrite memory with RL and reports
3.5M-token tasks; Recursive Language Models (Zhang, Kraska, Khattab, arXiv 2512.24601) keep the
context as a REPL variable the model queries programmatically and report no degradation past 10M
tokens, a broader and stronger claim than ours. Structural code indexes and agent-facing
code navigation are standard: SWE-agent's agent-computer interface (search/open/goto), Aider's
repo-map, and, as the brief describes them (not verified here), Codebase-Memory (arXiv 2603.27277)
and "Code Isn't Memory" (arXiv 2606.22417). As the brief describes them, TierMem (arXiv 2602.17913)
links a summary tier to an immutable raw tier by provenance and escalates to it, much as our ledger
links folds to spans; and Agent Zero Memory (arXiv 2608.29606) has a "citation lock" (cite only
evidence opened; abstain rather than guess), close to our Tier 0/1 grounding. Self-RAG and CRAG
critique and correct retrieval with a model; CoVe verifies with generated questions;
AbstentionBench evaluates abstention.

The narrow claims v4 can support, if the pilot and grid bear them out:

1. A **question-anchored answer gate** (target parsed from the question text alone, exact
   witness, stale-note supersession) whose refusals become **targeted repair** (the exact span of
   the asked target, or the latest note, put back in the window with the reason) rather than
   abstention or a model-judged correction. CRAG-style correction is model-scored; citation locks
   check that evidence was opened, not that it is about what was asked.
2. A **controlled scaling study** over rho spanning three orders of magnitude (C10 to C10000,
   effective rho 34 to 33,335 at W = 3,000), on one nested corpus, that separates abstention from
   fabrication and adds a forced-answer condition, with an index-only control (B5) and the v3
   arms run unchanged.
3. The window as a **cache over an exact, content-addressed store** with ledger invariants: every
   read hash-checked, every span re-readable exactly after eviction, every emitted answer's
   grounding re-checkable from the ledger (P1-P16), and the corpus level certified by a Merkle
   root in each run's configuration.

The index, the paging and the tools are engineering, not contributions.

## 6. Threats to validity

- **No live data.** Everything above was tested with a scripted fake model; whether a real model
  uses pre-resolution, repairs and the certificates as intended is what the pilot is for.
- **Templated questions favour pre-resolution.** Every corpus question names its target as a
  backticked qualified name the gate's parser reads, and the index keys follow the same module
  naming. That alignment is deliberate (the gate needs it) but makes lookups an upper bound on
  real-world addressability; a question that does not name its target gets no pre-resolution.
- **L4 vs B5 is not the gate alone.** L4 also differs by the per-episode ledger, folding and the
  ctx tools. L4-nogate (2.4b) isolates the gate: L4 vs L4-nogate. The gate also changes the
  conversation after a refusal (repairs add text), so later questions of the episode differ too.
- **L4 vs L mixes in the fixed question-folding bug** (finding 2). analyze-v4's primary L vs L4
  contrast (exploratory, decision 4) is on corpus questions, whose asked target is not a session
  note; every turn is the secondary one. A corpus question can still carry a note, so the bug is
  narrowed, not removed.
- **The 90-call episode bound** left 16% (allowed) and 40% (forced) of L's v3 rows unasked. v4
  runs every arm at 260 calls per episode (26 questions x 10, decision 1), so no question goes
  unasked for want of calls; L4 vs L (v3's 90) then also differs in the bound.
- **Fresh v4 items are partly unperturbed.** No v4 item overlaps a v3 item any run asked, but the
  rho-10 core is nearly exhausted, so 92 of the 234 corpus items are extras whose gold is an
  unperturbed literal (stat's octal modes, 0/1 mode constants): a model may know them, and only
  arms without a gate can answer from memory, which favours B3/B4 over L4 in accuracy. Rows carry
  `origin`; analyze-v4 reports accuracy by origin (exploratory).
- **The distractors are not CPython.** C10000's new files are third-party packages under
  Lib/site-packages/; the unchanged frame still says "the source tree of CPython 3.15". BM25 and
  the model may rank them as less relevant than more CPython would be, so the 1000 to 10000 step
  may stress retrieval less than "more of the same" would.
- **Token counts are conservative.** The corpus counts with calib-v1's upper envelope; point
  estimates are about 58% of those, so the effective rho is 20 to 19,403 rather than 34 to 33,335.
- **The pilot's v3 episodes at C10000:** locate answers stay unique and near-miss names absent,
  but 4 of the 16 session note values of episodes 11-12 (plain integers such as 2916) also occur
  in distractor files. The session questions stay unambiguous; questions v4 regenerates values
  absent from the whole extension.
- **Statistics.** analyze-v4 implements BUILD-V4B decision 5's confirmatory family, six tests per
  model with Holm within it (Haiku primary, Sonnet secondary): L4 non-inferior to B3 in accuracy
  (3 pp, item-paired, episode-cluster bootstrap), L4 fewer confident errors than B3 and than B4
  (exact McNemar), L4's accuracy flat over rho by TOST (+/-2 pp per decade) in both conditions, and
  forced at rho >= 100 L4 fewer unlabelled fabrications than B4. It is frozen only when PREREG-v4
  is. With 10 episodes per cell, equivalence may often be "not shown".
- The extension's sources come from this machine's site-packages. Since BUILD-V4B (decision 6)
  `corpus snapshot` copies every file the extension uses (the 8,425 extension files in the level's
  order, the base's 856 files and its manifest and perturbation records) into a content-addressed
  store, `data/corpus-src/objects/` (gitignored, 154 MB), and the committed
  `data/corpus-src/MANIFEST-cf9bb5aff4ed6.tsv` lists each with its sha256, size and source
  (`anyio==4.9.0@homebrew-python3.11`, `base:ca8e01d67009c`). `corpus extend` reads only the
  snapshot (a missing or altered object refuses) and refuses a build whose id is not the
  snapshot's. Rebuilt from the snapshot, cf9bb5aff4ed6 is reproduced byte for byte: the same id,
  all 9,281 files, perturbations.jsonl and index.sqlite identical, manifest.json identical but for
  its absolute `cache_dir`. A machine whose site-packages changed copies the objects instead.

## 7. Pilot and grid (not run)

From the main tree after `git merge --ff-only v4`, outside Claude Code (BUILD-V4B decision 7):

```
scripts/v4/pilot.sh    # snapshot, extension corpus from it, index (offline); then 2 live runs (6 arms, 3 rho), $2.50 cap each
scripts/v4/grid.sh     # Haiku: MODEL, EPISODES, RHOS, ARMS, MAXCALLS (260) default to the Haiku grid
MODEL=claude-sonnet-5-5 EPISODES=1-8 ARMS=B3,B4,B5,L,L4 scripts/v4/grid.sh
```

Both pass `--key research` and `--max-calls-episode 260`, take each run's exit status with zsh's
`${pipestatus[1]}`, flag FAILED/INCOMPLETE runs, keep each run's output in runs/v4/<id>.log, and
halt before a run whose --budget would take the COMBINED spend over every key's ledger (spent +
outstanding) past $180, the research key's usable cap. The grid runs one condition, one rho and
half of its episodes per run (all its arms), the first half everywhere before the second, Haiku
before Sonnet; it builds questions v4 (13 episodes) if missing, and stops unless the file's sha256
is PREREG-v4's and `questions overlap` finds no overlap. `plan --v4 --measured <pilot runs>`
prices the grids from the pilot and suggests each run's --budget.

## 8. Open questions

Answered by BUILD-V4B (Mahimn's decisions): the episode bound is 260 for every arm (1); questions
v4 use fresh items only (2); L4-nogate isolates the gate (3); L vs L4 is on corpus questions first
(4); the confirmatory family is six tests per model (5); the extension is built from a committed
source snapshot (6); the pilot and grid scripts (7); PREREG-v4 drafted for review (8). Still open:

1. The Sonnet grid's size against the remaining budget (section 4).
2. l-v3's question-folding bug stays frozen in L; no l-v3.2 is planned.
3. Freeze PREREG-v4 after the pilot fills its cost section.

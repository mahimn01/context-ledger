# PREREG-v4 Amendment 3: raise the research-key stop for the Sonnet L4-nogate arm

Dated Nov 5, 2026. Committed by Mahimn Patel before any Sonnet L4-nogate call under the new cap.
Everything not stated here is as frozen in PREREG-v4 (e903b45), Amendment 1 (0918dc1) and
Amendment 2 (this commit's parent, which added the Sonnet L4-nogate arm).

## Why, and when it was decided
Amendment 2 kept Amendment 1's soft stop of $150. At that commit the research key had spent
$122.345727, with about $27.65 left under $150. The arm's estimate x 1.2 was about $12.94 and its
16 run caps totalled $22.00, so it fitted. On Nov 5 at about 1:48 AM EST Mahimn raised the spend
authority for this work: the research key may be used up to its hard max of $200, including the
$20 reserve, so an overrun cannot cut the arm short and later paper-related work can use what is
left. No Sonnet L4-nogate call had been billed yet (`eval spend status` still $122.345727 spent,
$0 outstanding; no run id ending in -nogate1).

## Change
- Soft stop for scripts/v4/run-sonnet-nogate.sh: total research-key spend (spent + outstanding)
  stays at or under **$200** (the key's hard max). The $20 reserve is released for this arm
  (`--use-reserve` on each run).
- The six confirmatory tests are unchanged. The Sonnet L4-nogate comparisons stay secondary /
  exploratory as in Amendment 2.
- Per-run budgets stay as in Amendment 2 (2x expected). Balanced order, resume of incomplete
  runs, and halt-on-API-error are unchanged.
- If a run would push spent+outstanding+its budget past $200, halt and analyse the partial arm.

## Disclosure
Decided after Amendment 2 was written and after the confirmatory results in
runs/v4/grid-analysis-c260-fix1 were seen. No new confirmatory claim is added.

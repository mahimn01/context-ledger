"""Cost accounting derived purely from the ledger: per-call costs, totals, the cache hit ratio,
and each backstop-v1 fold's predicted saving next to an estimate of what it realized."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

from .backstop import s_cache
from .ids import frac_str, parse_frac
from .model import Event
from .pricing import PricingTable, usd


@dataclass(frozen=True)
class CostRow:
    call_id: str
    model: str
    input: int
    cache_write_5m: int
    cache_write_1h: int
    cache_read: int
    output: int
    cost_pusd: int


@dataclass(frozen=True)
class FoldSaving:
    """A backstop-v1 fold. `predicted_pusd` is S_cache from its decision record (before the risk
    term) and `J` the net objective; `realized_pusd` is an ESTIMATE from the following calls:
    calls x delta x p_r of reads avoided, minus the extra cache writes of the first call after
    the fold (beyond the call before it) at p_w - p_r."""

    fold_id: str
    zone: str
    delta: int
    rhat: int
    following_calls: int
    predicted_pusd: int
    J: Fraction
    realized_pusd: int


@dataclass(frozen=True)
class CostReport:
    rows: tuple[CostRow, ...]
    totals: dict
    cache_hit: Fraction  # read / (read + write + input)
    folds: tuple[FoldSaving, ...]

    def to_dict(self) -> dict:
        """Ints and strings only (dollars as display strings, ratios as "num/den")."""
        return {
            "rows": [{**row.__dict__, "cost_usd": usd(row.cost_pusd)} for row in self.rows],
            "totals": {**self.totals, "cost_usd": usd(self.totals["cost_pusd"])},
            "cache_hit": frac_str(self.cache_hit),
            "cache_hit_bp": int(self.cache_hit * 10000),
            "folds": [
                {
                    "fold_id": f.fold_id,
                    "zone": f.zone,
                    "delta": f.delta,
                    "rhat": f.rhat,
                    "following_calls": f.following_calls,
                    "predicted_pusd": f.predicted_pusd,
                    "predicted_usd": usd(f.predicted_pusd),
                    "J": frac_str(f.J),
                    "J_usd": usd(f.J),
                    "realized_estimate_pusd": f.realized_pusd,
                    "realized_estimate_usd": usd(f.realized_pusd),
                }
                for f in self.folds
            ],
        }

    def render(self) -> str:
        """A plain-text table for the CLI and the demo."""
        head = f"{'call':<22} {'model':<18} {'input':>7} {'write5m':>8} {'write1h':>8} {'read':>7} {'output':>7} {'cost':>11}"
        lines = [head, "-" * len(head)]
        for r in self.rows:
            lines.append(f"{r.call_id:<22} {r.model:<18} {r.input:>7} {r.cache_write_5m:>8} {r.cache_write_1h:>8} {r.cache_read:>7} {r.output:>7} {usd(r.cost_pusd):>11}")
        t = self.totals
        lines.append(f"{'total':<22} {'':<18} {t['input']:>7} {t['cache_write_5m']:>8} {t['cache_write_1h']:>8} {t['cache_read']:>7} {t['output']:>7} {usd(t['cost_pusd']):>11}")
        lines.append(f"cache hit ratio (read / all input): {float(self.cache_hit):.1%}")
        for f in self.folds:
            lines.append(
                f"fold {f.fold_id} ({f.zone}): delta {f.delta} tokens, R-hat {f.rhat}; predicted saving {usd(f.predicted_pusd)}"
                f" (J {usd(f.J)}); realized over {f.following_calls} calls, estimate: {usd(f.realized_pusd)}"
            )
        return "\n".join(lines)


def _writes(usage: dict) -> int:
    return usage["cache_creation_input_tokens"]


def cost_report(events: Sequence[Event], pricing: PricingTable | None = None) -> CostReport:
    """Cost of every model_call, priced with `pricing` if given, else with the table in force
    at that point of the ledger (which P16 guarantees reproduces the recorded cost)."""
    in_force: dict | None = None
    rows: list[CostRow] = []
    calls: list[tuple[int, dict]] = []  # (seq, usage) of each model_call
    folds: list[tuple[int, dict]] = []  # (seq, fold) of each fold
    tables: dict[str, PricingTable] = {}
    for event in events:
        p = event.payload
        if event.type == "pricing_set":
            in_force = p["pricing"]
            tables[p["pricing_hash"]] = PricingTable.from_dict(p["pricing"])
        elif event.type == "fold":
            folds.append((event.seq, p["fold"]))
        elif event.type == "model_call":
            table = pricing or PricingTable.from_dict(in_force)
            usage = p["usage"]
            cost = table.cost_pusd(p["model"], usage, p["build"]["ttl"])
            rows.append(CostRow(p["call_id"], p["model"], usage["input_tokens"], usage["cache_creation_5m"], usage["cache_creation_1h"],
                                usage["cache_read_input_tokens"], usage["output_tokens"], cost))  # fmt: skip
            calls.append((event.seq, usage))
    keys = ("input", "cache_write_5m", "cache_write_1h", "cache_read", "output", "cost_pusd")
    totals = {k: sum(getattr(r, k) for r in rows) for k in keys}
    seen = totals["input"] + totals["cache_write_5m"] + totals["cache_write_1h"] + totals["cache_read"]
    hit = Fraction(totals["cache_read"], seen) if seen else Fraction(0)
    savings = []
    for n, (seq, fold) in enumerate(folds):
        policy = fold.get("policy") or {}
        if policy.get("id") != "backstop-v1":
            continue
        params = policy["params"]
        price = (pricing or tables[policy["pricing_hash"]]).price(params["model"])
        p_w, p_r = price.cache_write(params["ttl"]), price.cache_read
        until = folds[n + 1][0] if n + 1 < len(folds) else None
        after = [u for s, u in calls if s > seq and (until is None or s < until)]
        before = [u for s, u in calls if s < seq]
        extra_writes = max(0, _writes(after[0]) - (_writes(before[-1]) if before else 0)) if after else 0
        realized = len(after) * policy["delta"] * p_r - extra_writes * (p_w - p_r)
        predicted = s_cache(policy["rhat"], policy["delta"], policy["L"], policy["P_i"], p_w, p_r, 0, policy["cache_cold"])
        savings.append(FoldSaving(fold["fold_id"], policy["zone"], policy["delta"], policy["rhat"], len(after), predicted,
                                  parse_frac(policy["J"]), realized))  # fmt: skip
    return CostReport(tuple(rows), totals, hit, tuple(savings))

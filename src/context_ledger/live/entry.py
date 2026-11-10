"""open_live: the one way into live mode. Every live command calls it, and it checks, in this
order, before the key is read or a client is built:

1. the live gate (the --live flag and a safe environment);
2. a verified price table;
3. a SpendGuard that can lock and persist its ledger (it fails closed);
4. the key provider returns a key, which is registered for redaction everywhere.

The key provider's profile (research by default, or aperture) picks the SpendGuard's ledger and
caps (spend.KEY_LEDGERS): no caller passes a ledger, so one key is never charged against the
other's ledger.

Only then does the live factory build the SDK client, which is wrapped in a GuardedClient.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from ..pricing import PricingTable
from .factory import make_live_client
from .gate import LiveRefused, require_live
from .guarded import GuardedClient
from .keys import KeyProvider, Secret
from .redact import install_log_redaction, register_secret
from .spend import KEY_LEDGERS, SpendGuard

SPEND_DIR = Path(__file__).resolve().parents[3] / ".clp-spend"


@dataclass
class LiveContext:
    client: GuardedClient
    guard: SpendGuard
    secret: Secret

    def close(self) -> None:
        self.guard.close()


def open_live(
    args: object,
    environ: Mapping[str, str],
    *,
    keys: KeyProvider,
    pricing: PricingTable,
    run_id: str,
    run_cap_usd: str | None,
    use_reserve: bool = False,
    global_cap_usd: str | int | None = None,
    spend_dir: str | Path = SPEND_DIR,
    estimator=None,
    factory: Callable[[Secret], object] = make_live_client,
) -> LiveContext:
    """Pass every check of the live gate and return the guarded live client (see above)."""
    require_live(args, environ)
    if not pricing.verified:
        raise LiveRefused(f"price table {pricing.version} is not verified")
    key = KEY_LEDGERS[keys.profile]
    guard = SpendGuard(key.path(spend_dir), run_id=run_id, pricing=pricing, key=key, use_reserve=use_reserve, global_cap_usd=global_cap_usd,
                       run_cap_usd=run_cap_usd)  # fmt: skip
    try:
        install_log_redaction()
        secret = keys.get()
        register_secret(secret)
        client = GuardedClient(factory(secret), guard, estimator=estimator)
    except BaseException:
        guard.close()
        raise
    return LiveContext(client, guard, secret)
